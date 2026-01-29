import torch
from sparsetriton.tensor import SparseTensor
from sparsetriton.nn.functional.spatial import sparse_pooling
import random

def debug_sparse_pooling(mode='avg'):
    print(f"Debugging sparse_pooling with mode='{mode}'")

    # 1. 더미 SparseTensor 데이터 생성
    N_IN = 10  # 입력 비제로(non-zero) 원소 수
    C_FEAT = 32 # 특징 채널 수
    SPATIAL_DIM = 5 # 공간 차원 (예: 5x5x5 그리드)
    BATCH_SIZE = 1

    # Features: 무작위 데이터
    F = torch.randn(N_IN, C_FEAT, device='cuda', requires_grad=True)

    # Coordinates: 공간 차원 내에서 무작위 고유 좌표 생성 (batch_idx, x, y, z)
    # 첫 번째 컬럼은 배치 인덱스 (0), 다음 세 컬럼은 공간 좌표
    coords_set = set()
    C_list = []
    while len(C_list) < N_IN:
        b_idx = 0
        x = random.randint(0, SPATIAL_DIM - 1)
        y = random.randint(0, SPATIAL_DIM - 1)
        z = random.randint(0, SPATIAL_DIM - 1)
        coord_tuple = (b_idx, x, y, z)
        if coord_tuple not in coords_set:
            coords_set.add(coord_tuple)
            C_list.append(list(coord_tuple))
    
    C = torch.tensor(C_list, dtype=torch.int32, device='cuda')

    spatial_shape = (SPATIAL_DIM, SPATIAL_DIM, SPATIAL_DIM)

    input_sparse_tensor = SparseTensor(F, C, spatial_shape=spatial_shape, batch_size=BATCH_SIZE)

    # 풀링 파라미터
    kernel_size = 2
    stride = 2
    padding = 0

    print(f"Input SparseTensor:")
    print(f"  Features shape: {input_sparse_tensor.F.shape}")
    print(f"  Coordinates shape: {input_sparse_tensor.C.shape}")
    print(f"  Spatial shape: {input_sparse_tensor.spatial_shape}")
    print(f"  Batch size: {input_sparse_tensor.batch_size}")
    print(f"Pooling parameters: kernel_size={kernel_size}, stride={stride}, padding={padding}, mode='{mode}'")

    # 2. sparse_pooling 호출
    try:
        output_sparse_tensor = sparse_pooling(
            input_sparse_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            mode=mode
        )

        print("\nForward pass successful!")
        print(f"Output SparseTensor:")
        print(f"  Features shape: {output_sparse_tensor.F.shape}")
        print(f"  Coordinates shape: {output_sparse_tensor.C.shape}")
        print(f"  Spatial shape: {output_sparse_tensor.spatial_shape}")
        print(f"  Batch size: {output_sparse_tensor.batch_size}")

        # 3. 역전파 수행
        # 더미 손실 생성
        loss = output_sparse_tensor.F.sum()
        print("\nStarting backward pass...")
        loss.backward()
        print("Backward pass successful!")
        
        # 입력 특징에 대한 기울기 확인
        if F.grad is not None:
            print(f"Gradient for input features (F.grad) shape: {F.grad.shape}")
        else:
            print("F.grad is None.")

    except Exception as e:
        print(f"\nAn error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 기본적으로 'avg' 모드를 테스트합니다.
    debug_sparse_pooling(mode='avg')
    # 최대 풀링도 테스트하려면 아래 주석을 해제하세요.
    # debug_sparse_pooling(mode='max')
