from ast import Tuple
from shutil import make_archive
import torch
from sparsetriton import SparseTensor
import triton
import triton.language as tl

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
        in_coords_ptr, in_feats_ptr,
        out_coords_ptr, out_feats_ptr,
        N_IN, N_OUT, # Number of input/output non-zero elements
        BLOCK_SIZE: tl.constexpr,
        # TODO: Add other necessary parameters like kernel_size, stride, etc.
        # This is where the core logic will go.
    ):
        pass # Placeholder for Triton JIT kernel

    @staticmethod
    def forward(ctx, input_tensor: SparseTensor, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
        # TODO: Implement actual data preparation and kernel launch
        # For now, just save inputs and return a dummy SparseTensor
        
        # Example of saving for backward pass (adjust based on actual needs)
        ctx.save_for_backward(input_tensor.F, input_tensor.C) # Save features and coordinates
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        # ... save other parameters

        # Placeholder for output
        output_feats = torch.empty((0, input_tensor.F.shape[1]), device=input_tensor.F.device, dtype=input_tensor.F.dtype)
        output_coords = torch.empty((0, 4), device=input_tensor.C.device, dtype=input_tensor.C.dtype)
        
        # Example kernel call (will need actual implementation)
        # N_IN = input_tensor.F.shape[0]
        # _sparse_pooling_forward_kernel[N_IN](
        #    input_tensor.C, input_tensor.F,
        #    output_coords, output_feats,
        #    N_IN, N_OUT, # N_OUT would be determined by the pooling logic
        # )

        return SparseTensor(output_feats, output_coords, spatial_shape=(1,1,1), batch_size=1) # Dummy return

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        ],
        key=['N_OUT'], # Key for autotuning. N_OUT would be number of output non-zero elements.
    )
    @triton.jit
    def _sparse_pooling_backward_kernel(
        grad_out_feats_ptr, # Gradient from subsequent layer
        in_coords_ptr, # Original input coordinates
        grad_in_feats_ptr, # Gradient to propagate back to input features
        N_OUT, N_IN, # Number of output/input non-zero elements
        BLOCK_SIZE: tl.constexpr,
        # TODO: Add other necessary parameters
    ):
        pass # Placeholder for Triton JIT kernel

    @staticmethod
    def backward(ctx, grad_output: SparseTensor):
        # TODO: Implement actual data preparation and kernel launch for backward
        input_feats, input_coords = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        # ... retrieve other parameters

        grad_input_feats = torch.zeros_like(input_feats)
        grad_input_coords = None # Coordinates typically not differentiated

        # Example kernel call (will need actual implementation)
        # N_OUT = grad_output.F.shape[0]
        # _sparse_pooling_backward_kernel[N_OUT](
        #    grad_output.F,
        #    input_coords,
        #    grad_input_feats,
        #    N_OUT, N_IN,
        # )

        return grad_input_feats, None, None, None, None, None, None # Return gradients for each input to forward


def sparse_pooling(input: SparseTensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> SparseTensor:
    """
    Placeholder for sparse pooling operation.
    """
    return SparsePoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

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
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 1, in_x + dx, mask=out_mask)
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 2, in_y + dy, mask=out_mask)
                            tl.store(out_coords_ptr + curr_out_offs * 4 + 3, in_z + dz, mask=out_mask)
                        
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
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
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

        for c in range(0, C, BLOCK_SIZE_N):
            c_off = c + tl.arange(0, BLOCK_SIZE_N)
            c_mask = c_off < C
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_N), dtype=tl.)

            for step in range(prod_scale):
                pass

            tl.store(
                grad_in_feats_ptr + in_offs[:, None] * C + c_off[None, :], 
                acc.to(grad_in_feats_ptr.dtype.element_ty), 
                mask = in_mask[:, None] & c_mask[None, :]
            )


            
            

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        N_OUT, C = grad_output.shape
        N_IN = ctx.N_IN
        scale_factor = ctx.scale_factor

        grad_input_feats = torch.zeros((N_IN, C), device=grad_output.device, dtype=grad_output.dtype)
        grad_input_coords = None

        return grad_input_feats, None, None, None


def sparse_upsample(input: SparseTensor, size=None, scale_factor=None) -> SparseTensor:
    """
    Placeholder for sparse upsample operation.
    """
    output_feats, output_coords, new_spatial_shape = SparseUpsampleFunction.apply(input.F, input.C, input.spatial_shape, scale_factor)
    return SparseTensor(
        output_feats, output_coords, new_spatial_shape
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

