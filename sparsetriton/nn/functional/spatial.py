from ast import Tuple
from shutil import make_archive
import torch
from sparsetriton import SparseTensor
import triton
import triton.language as tl
from sparsetriton.utils.hash import flatten_coord, unflatten_coord

# --- Sparse Pooling ---
class SparsePoolingFunction(torch.autograd.Function):
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['N_IN'], # Key for autotuning. N_IN would be number of input non-zero elements.
    )
    @triton.jit
    def _sparse_pooling_forward_kernel(
        in_feats_ptr, out_feats_ptr,
        map_ptr, count_ptr,
        N_IN, C,
        MODE: tl.constexpr, # 0: max, 1: avg
        BLOCK_SIZE: tl.constexpr,
        BLOCK_C: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_IN
        
        out_idx = tl.load(map_ptr + offs, mask=mask, other=-1)
        active = (out_idx != -1) & mask
        
        in_base_ptr = in_feats_ptr + offs[:, None] * C
        out_base_ptr = out_feats_ptr + out_idx[:, None] * C
        
        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C
            op_mask = active[:, None] & c_mask[None, :]
            
            val = tl.load(in_base_ptr + c_offs[None, :], mask=op_mask, other=0.0)
            target_ptr = out_base_ptr + c_offs[None, :]
            
            if MODE == 0: # MAX
                tl.atomic_max(target_ptr, val, mask=op_mask)
            elif MODE == 1: # AVG
                tl.atomic_add(target_ptr, val, mask=op_mask)

    @staticmethod
    def forward(ctx, feats, coords, spatial_shape, batch_size, kernel_size, stride, padding, mode='max'):
        if stride is None:
            stride = kernel_size
            
        # coords = input_tensor.C
        device = coords.device
        
        # Map inputs to outputs (Downsampling logic)
        spatial_coords = coords[:, 1:]
        out_spatial = (spatial_coords + padding) // stride
        out_coords_full = torch.cat([coords[:, :1], out_spatial], dim=1)
        flat_coords = flatten_coord(out_coords_full)
        unique_flat, inverse_indices = torch.unique(flat_coords, sorted=True, return_inverse=True)
        unique_coords = unflatten_coord(unique_flat)
        
        N_IN, C = feats.shape
        N_OUT = unique_coords.shape[0]
        out_feats = torch.empty((N_OUT, C), device=device, dtype=feats.dtype)

        mode_const = 0
        if mode == 'max':
            out_feats.fill_(float('-inf'))
            mode_const = 0
        elif mode == 'avg':
            out_feats.fill_(0.0)
            mode_const = 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        # Optimization: Use torch.bincount instead of atomic_add in kernel
        if mode == 'avg':
            count = torch.bincount(inverse_indices, minlength=N_OUT).to(feats.dtype).view(-1, 1).clamp(min=1)
        else:
            count = None
        
        grid = lambda meta: (triton.cdiv(N_IN, meta['BLOCK_SIZE']), )
        BLOCK_C = 64
        if C < 64: BLOCK_C = triton.next_power_of_2(C)
        
        SparsePoolingFunction._sparse_pooling_forward_kernel[grid](
            in_feats_ptr=feats,
            out_feats_ptr=out_feats,
            map_ptr=inverse_indices,
            count_ptr=count if count is not None else inverse_indices,
            N_IN=N_IN, C=C,
            MODE=mode_const,
            BLOCK_C=BLOCK_C
        )
        
        if mode == 'avg':
            out_feats = out_feats / count
            
        ctx.save_for_backward(feats, coords, inverse_indices, out_feats, count)
        ctx.mode = mode
        ctx.N_IN = N_IN
        
        new_spatial_shape = tuple((s + 2*padding - kernel_size)//stride + 1 for s in spatial_shape)
        
        return out_feats, unique_coords, new_spatial_shape, batch_size

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['N_IN'],
    )
    @triton.jit
    def _sparse_pooling_backward_kernel(
        grad_out_feats_ptr, # Gradient from subsequent layer
        grad_in_feats_ptr, # Gradient to propagate back to input features
        in_feats_ptr, out_feats_ptr,
        map_ptr, count_ptr,
        N_IN, C,
        MODE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_C: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_IN
        
        out_idx = tl.load(map_ptr + offs, mask=mask, other=-1)
        active = (out_idx != -1) & mask
        
        in_base_ptr = in_feats_ptr + offs[:, None] * C
        grad_in_base_ptr = grad_in_feats_ptr + offs[:, None] * C
        out_base_ptr = out_feats_ptr + out_idx[:, None] * C
        grad_out_base_ptr = grad_out_feats_ptr + out_idx[:, None] * C

        if MODE == 1: # AVG
            cnt = tl.load(count_ptr + out_idx, mask=active, other=1.0)[:, None]
        
        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C
            op_mask = active[:, None] & c_mask[None, :]

            grad_out = tl.load(grad_out_base_ptr + c_offs[None, :], mask=op_mask, other=0.0)
            grad_val = 0.0
            
            if MODE == 0: # MAX
                in_val = tl.load(in_base_ptr + c_offs[None, :], mask=op_mask, other=0.0)
                out_val = tl.load(out_base_ptr + c_offs[None, :], mask=op_mask, other=0.0)
                is_max = (in_val == out_val)
                grad_val = tl.where(is_max, grad_out, 0.0)
            elif MODE == 1: # AVG
                grad_val = grad_out / cnt
                
            tl.store(grad_in_base_ptr + c_offs[None, :], grad_val, mask=op_mask)

    @staticmethod
    def backward(ctx, grad_out_feats, grad_out_coords, grad_spatial_shape, grad_batch_size):
        in_feats, in_coords, map_tensor, out_feats, count = ctx.saved_tensors
        mode = ctx.mode
        N_IN = ctx.N_IN
        C = in_feats.shape[1]
        
        grad_in = torch.zeros_like(in_feats)
        
        grid = lambda meta: (triton.cdiv(N_IN, meta['BLOCK_SIZE']), )
        BLOCK_C = 64
        if C < 64: BLOCK_C = triton.next_power_of_2(C)
        
        mode_const = 0 if mode == 'max' else 1
        
        SparsePoolingFunction._sparse_pooling_backward_kernel[grid](
            grad_out_feats_ptr=grad_out_feats,
            grad_in_feats_ptr=grad_in,
            in_feats_ptr=in_feats,
            out_feats_ptr=out_feats,
            map_ptr=map_tensor,
            count_ptr=count if count is not None else map_tensor,
            N_IN=N_IN, C=C,
            MODE=mode_const,
            BLOCK_C=BLOCK_C
        )
        
        return grad_in, None, None, None, None, None, None, None


def sparse_pooling(input: SparseTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, mode='max') -> SparseTensor:
    out_feats, out_coords, new_spatial_shape, batch_size = SparsePoolingFunction.apply(
        input.F, input.C, input.spatial_shape, input.batch_size,
        kernel_size, stride, padding, mode
    )
    return SparseTensor(out_feats, out_coords, spatial_shape=new_spatial_shape, batch_size=batch_size)

# --- Sparse Upsample ---
class SparseUpsampleFunction(torch.autograd.Function):
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=["scale_factor"],
        cache_results=True
    )
    @triton.jit
    def _sparse_upsample_forward_kernel(
        in_coords_ptr, in_feats_ptr,
        out_coords_ptr, out_feats_ptr,
        N_IN, C, N_OUT, # N_IN, C, N_OUT은 일반 변수
        scale_factor: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_C: tl.constexpr  # [추가] 한 번에 처리할 채널 덩어리 크기 (예: 32 or 64)
    ):
        pid = tl.program_id(0)
        
        # stride 미리 계산
        s = scale_factor
        prod_scale = s * s * s
        
        in_start = pid * BLOCK_SIZE

        in_offs = in_start + tl.arange(0, BLOCK_SIZE)
        n_mask = in_offs < N_IN

        in_b = tl.load(in_coords_ptr + (in_offs * 4) + 0, mask=n_mask, other=0)
        in_x = tl.load(in_coords_ptr + (in_offs * 4) + 1, mask=n_mask, other=0)
        in_y = tl.load(in_coords_ptr + (in_offs * 4) + 2, mask=n_mask, other=0)
        in_z = tl.load(in_coords_ptr + (in_offs * 4) + 3, mask=n_mask, other=0)
        
        for c_start in range(0, C, BLOCK_C):
            c_offs = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C
            
            load_mask = n_mask[:, None] & c_mask[None, :]
            
            feat_chunk = tl.load(
                in_feats_ptr + (in_offs[:, None] * C + c_offs[None, :]),
                mask=load_mask,
                other=0.0
            )

            step = 0
            for dx in range(scale_factor):
                for dy in range(scale_factor):
                    for dz in range(scale_factor):
                        curr_out_offs = in_offs * prod_scale + step
                        
                        out_mask = (curr_out_offs < N_OUT) & n_mask
                        
                        store_mask = out_mask[:, None] & c_mask[None, :]
                        tl.store(
                            out_feats_ptr + (curr_out_offs[:, None] * C + c_offs[None, :]),
                            feat_chunk,
                            mask=store_mask
                        )
                        
                        if c_start == 0:
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 0, in_b, mask=out_mask)
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 1, in_x * scale_factor + dx, mask=out_mask)
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 2, in_y * scale_factor + dy, mask=out_mask)
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 3, in_z * scale_factor + dz, mask=out_mask)
                        
                        step += 1

    @staticmethod
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor, spatial_shape: Tuple, scale_factor:int):
        # TODO: Implement actual data preparation and kernel launch
        feats, coords = feats.contiguous(), coords.contiguous()
        ctx.scale_factor = scale_factor
        N, C = feats.shape
        _, D = coords.shape
        N_OUT = N *  (scale_factor ** 3)
        ctx.N_IN = N
        ctx.N_OUT = N_OUT

        output_feats = torch.empty((N_OUT, C), device=feats.device, dtype=feats.dtype)
        output_coords = torch.empty((N_OUT, 4), device=coords.device, dtype=coords.dtype)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
        
        BLOCK_C = 64
        if C < 64: BLOCK_C = triton.next_power_of_2(C)

        SparseUpsampleFunction._sparse_upsample_forward_kernel[grid](
            in_coords_ptr=coords,
            in_feats_ptr=feats,
            out_coords_ptr=output_coords,
            out_feats_ptr=output_feats,
            scale_factor=scale_factor,
            N_IN = N, C=C, N_OUT=N_OUT, BLOCK_C=BLOCK_C
        )
        new_spatial_shape = (s * scale_factor for s in spatial_shape)
        return output_feats, output_coords, new_spatial_shape

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_N': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE_N': 256}, num_warps=8),
        ],
        key=['scale_factor'],
    )
    @triton.jit
    def _sparse_upsample_backward_kernel(
        grad_out_feats_ptr, # (N_OUT, C)
        grad_in_feats_ptr, # (N_IN, C)
        scale_factor: tl.constexpr,
        N_OUT, N_IN, C,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        in_offs = BLOCK_SIZE_N * pid_n + tl.arange(0, BLOCK_SIZE_N)
        in_mask = in_offs < N_IN
        prod_scale = scale_factor * scale_factor * scale_factor

        for c in range(0, C, BLOCK_SIZE_C):
            c_off = c + tl.arange(0, BLOCK_SIZE_C)
            c_mask = c_off < C
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=grad_in_feats_ptr.dtype.element_ty)
            for step in range(prod_scale):
                out_offs = in_offs * prod_scale + step
                out_mask = out_offs < N_OUT
                acc += tl.load(
                    grad_out_feats_ptr + out_offs[:, None] * C + c_off[None, :], 
                    mask = out_mask[:, None] & c_mask,
                    other = 0
                )
            tl.store(
                grad_in_feats_ptr + in_offs[:, None] * C + c_off[None, :], 
                acc,
                mask = in_mask[:, None] & c_mask[None, :]
            )

            

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, output_coords, new_spatial_shape):
        grad_output = grad_output.contiguous()
        N_OUT, C = grad_output.shape
        N_IN = ctx.N_IN
        scale_factor = ctx.scale_factor

        grad_input_feats = torch.zeros((N_IN, C), device=grad_output.device, dtype=grad_output.dtype)
        grad_input_coords = None
        grid = lambda meta: (triton.cdiv(N_IN, meta['BLOCK_SIZE_N']), )
        BLOCK_SIZE_C = 64
        if C < 64: BLOCK_SIZE_C = triton.next_power_of_2(C)
        SparseUpsampleFunction._sparse_upsample_backward_kernel[grid](
            grad_out_feats_ptr=grad_output,
            grad_in_feats_ptr=grad_input_feats,
            scale_factor=scale_factor,
            N_OUT=N_OUT, N_IN=N_IN, C=C, 
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )

        return grad_input_feats, None, None, None


def sparse_upsample(input: SparseTensor, size=None, scale_factor=None) -> SparseTensor:
    """
    Placeholder for sparse upsample operation.
    """
    output_feats, output_coords, new_spatial_shape = SparseUpsampleFunction.apply(input.F, input.C, input.spatial_shape, scale_factor)
    return SparseTensor(
        output_feats, output_coords, new_spatial_shape, batch_size=input.batch_size
    )

# --- Sparse Downsample ---
class SparseDownsampleFunction(torch.autograd.Function):
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['N_IN'],
    )
    @triton.jit
    def _sparse_downsample_forward_kernel(
        in_coords_ptr, in_feats_ptr,
        out_coords_ptr, out_feats_ptr,
        N_IN, N_OUT, # N_OUT would be determined by downsampling logic
        BLOCK_SIZE: tl.constexpr,
        # TODO: Add other necessary parameters
    ):
        pass

    @staticmethod
    def forward(ctx, input_tensor: SparseTensor, size, scale_factor, mode, align_corners):
        # TODO: Implement actual data preparation and kernel launch
        ctx.save_for_backward(input_tensor.F, input_tensor.C)
        ctx.size = size
        ctx.scale_factor = scale_factor
        ctx.mode = mode
        # ... save other parameters

        output_feats = torch.empty((0, input_tensor.F.shape[1]), device=input_tensor.F.device, dtype=input_tensor.F.dtype)
        output_coords = torch.empty((0, 4), device=input_tensor.C.device, dtype=input_tensor.C.dtype)
        
        return SparseTensor(output_feats, output_coords, spatial_shape=(1,1,1), batch_size=1)

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['N_OUT'],
    )
    @triton.jit
    def _sparse_downsample_backward_kernel(
        grad_out_feats_ptr,
        in_coords_ptr,
        grad_in_feats_ptr,
        N_OUT, N_IN,
        BLOCK_SIZE: tl.constexpr,
        # TODO: Add other necessary parameters
    ):
        pass

    @staticmethod
    def backward(ctx, grad_output: SparseTensor):
        # TODO: Implement actual data preparation and kernel launch for backward
        input_feats, input_coords = ctx.saved_tensors
        size = ctx.size
        scale_factor = ctx.scale_factor
        mode = ctx.mode

        grad_input_feats = torch.zeros_like(input_feats)
        grad_input_coords = None

        return grad_input_feats, None, None, None, None
    
def sparse_downsample():
    raise NotImplementedError
