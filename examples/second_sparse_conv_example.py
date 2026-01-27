import torch
import torch.nn as nn
from sparsetriton import SparseTensor
from sparsetriton.nn.modules.conv import Conv3d
from sparsetriton.nn.modules.norm import SparseBatchNorm
from sparsetriton.nn.modules.activation import ReLU
import os
from typing import Tuple, List

# 실제 데이터셋 로딩 및 전처리를 위한 도우미 함수
def load_oakland_sparse_data(
    file_path: str,
    spatial_shape: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float],
    point_cloud_range: List[float], # [min_x, max_x, min_y, max_y, min_z, max_z]
    in_channels: int,
    batch_idx: int,
):
    """
    Oakland 3D 데이터셋의 .xyz_label_conf 파일을 읽고 SparseTensor 형식으로 변환합니다.
    Args:
        file_path: .xyz_label_conf 파일의 경로.
        spatial_shape: (D, H, W) 형태의 3D 공간 크기 (복셀 단위).
        voxel_size: (vx, vy, vz) 형태의 각 복셀의 실제 세계 크기 (미터).
        point_cloud_range: [min_x, max_x, min_y, max_y, min_z, max_z] 형태의 포인트 클라우드 범위.
        in_channels: 각 포인트의 특징 채널 수 (예: x, y, z, intensity).
        batch_idx: 이 데이터 샘플의 배치 인덱스.
    Returns:
        SparseTensor: 특징(feats)과 좌표(coords)를 포함하는 희소 텐서.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # .xyz_label_conf 파일에서 포인트 로드
    # 파일 형식: x y z intensity label
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: # 최소한 x, y, z, intensity가 있어야 함
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    intensity = float(parts[3])
                    points.append([x, y, z, intensity])
                except ValueError:
                    # 파싱 오류가 있는 줄은 건너뜜니다.
                    continue
    
    if not points:
        print(f"경고: {file_path}에서 유효한 포인트가 발견되지 않았습니다.")
        return SparseTensor(
            feats=torch.empty(0, in_channels, device=device),
            coords=torch.empty(0, 4, dtype=torch.int32, device=device), # batch_idx, d, h, w
            spatial_shape=spatial_shape,
            batch_size=1 # 현재는 단일 샘플
        )

    points = torch.tensor(points, dtype=torch.float32, device=device)

    # 포인트 클라우드 범위 필터링
    mask = (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] < point_cloud_range[1]) & \
           (points[:, 1] >= point_cloud_range[2]) & (points[:, 1] < point_cloud_range[3]) & \
           (points[:, 2] >= point_cloud_range[4]) & (points[:, 2] < point_cloud_range[5])
    
    points = points[mask]

    if points.shape[0] == 0:
        print(f"경고: 범위 필터링 후 {file_path}에 포인트가 없습니다.")
        return SparseTensor(
            feats=torch.empty(0, in_channels, device=device),
            coords=torch.empty(0, 4, dtype=torch.int32, device=device),
            spatial_shape=spatial_shape,
            batch_size=1
        )

    # 연속 좌표를 복셀 좌표로 변환
    # (x - min_range) / voxel_size
    voxel_coords = torch.empty_like(points[:, :3], dtype=torch.int32, device=device)
    voxel_coords[:, 0] = torch.floor((points[:, 0] - point_cloud_range[0]) / voxel_size[0])
    voxel_coords[:, 1] = torch.floor((points[:, 1] - point_cloud_range[2]) / voxel_size[1])
    voxel_coords[:, 2] = torch.floor((points[:, 2] - point_cloud_range[4]) / voxel_size[2])

    # 복셀 좌표가 spatial_shape 범위 내에 있는지 확인
    mask = (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < spatial_shape[0]) & \
           (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < spatial_shape[1]) & \
           (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < spatial_shape[2])
    
    points = points[mask]
    voxel_coords = voxel_coords[mask]

    if points.shape[0] == 0:
        print(f"경고: 공간 범위 필터링 후 {file_path}에 포인트가 없습니다.")
        return SparseTensor(
            feats=torch.empty(0, in_channels, device=device),
            coords=torch.empty(0, 4, dtype=torch.int32, device=device),
            spatial_shape=spatial_shape,
            batch_size=1
        )

    # 고유 복셀 찾기 및 특징 집계 (평균 강도)
    # (d, h, w) -> (sum_intensity, count, sum_x, sum_y, sum_z)
    unique_voxel_map = {}
    for i in range(voxel_coords.shape[0]):
        coord_tuple = tuple(voxel_coords[i].tolist()) # (d, h, w)
        intensity_val = points[i, 3].item()
        
        # 특징으로 x, y, z 좌표를 사용하므로, 이를 저장
        feat_x, feat_y, feat_z = points[i, 0].item(), points[i, 1].item(), points[i, 2].item()
        
        if coord_tuple not in unique_voxel_map:
            unique_voxel_map[coord_tuple] = [intensity_val, 1, feat_x, feat_y, feat_z]
        else:
            unique_voxel_map[coord_tuple][0] += intensity_val
            unique_voxel_map[coord_tuple][1] += 1
            unique_voxel_map[coord_tuple][2] += feat_x
            unique_voxel_map[coord_tuple][3] += feat_y
            unique_voxel_map[coord_tuple][4] += feat_z


    final_coords = []
    final_feats = []

    for coord_tuple, data in unique_voxel_map.items():
        sum_intensity, count, sum_x, sum_y, sum_z = data
        avg_intensity = sum_intensity / count
        avg_x, avg_y, avg_z = sum_x / count, sum_y / count, sum_z / count
        
        final_coords.append(list(coord_tuple))
        if in_channels == 4:
            final_feats.append([avg_x, avg_y, avg_z, avg_intensity])
        elif in_channels == 3:
            final_feats.append([avg_x, avg_y, avg_z])
        else:
            final_feats.append(torch.randn(in_channels, device=device).tolist()) # 다른 채널 수에 대한 더미

    if not final_coords:
        print(f"경고: 고유 복셀 집계 후 {file_path}에 유효한 데이터가 없습니다.")
        return SparseTensor(
            feats=torch.empty(0, in_channels, device=device),
            coords=torch.empty(0, 4, dtype=torch.int32, device=device),
            spatial_shape=spatial_shape,
            batch_size=1
        )

    final_coords_tensor = torch.tensor(final_coords, dtype=torch.int32, device=device)
    final_feats_tensor = torch.tensor(final_feats, dtype=torch.float32, device=device)

    # 배치 인덱스 추가 (D, H, W) -> (B, D, H, W)
    batch_tensor = torch.full((final_coords_tensor.shape[0], 1), batch_idx, dtype=torch.int32, device=device)
    coords_with_batch = torch.cat([batch_tensor, final_coords_tensor], dim=1)

    return SparseTensor(feats=final_feats_tensor, coords=coords_with_batch, spatial_shape=spatial_shape, batch_size=1)


# 가상의 데이터 생성을 위한 도우미 함수 (실제 데이터셋은 더 복잡합니다)
def generate_realistic_sparse_data(batch_size: int, spatial_shape: Tuple[int, int, int], in_channels: int, target_num_voxels: int):
    """
    SECOND 모델의 입력과 유사한 보다 사실적인 희소 3D 데이터(포인트 클라우드에서 복셀화된)를 생성합니다.
    몇 개의 객체와 배경 노이즈를 시뮬레이션합니다.
    Args:
        batch_size: 배치 크기.
        spatial_shape: (D, H, W) 형태의 3D 공간 크기 (복셀 단위).
        in_channels: 각 포인트의 특징 채널 수 (예: x, y, z, intensity).
        target_num_voxels: 각 배치 항목당 생성하려는 목표 활성 복셀 수.
    Returns:
        SparseTensor: 특징(feats)과 좌표(coords)를 포함하는 희소 텐서.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_coords = []
    all_feats = []

    for b in range(batch_size):
        scene_points = []
        scene_intensities = []

        # 1. 객체 생성 (예: 작은 큐브)
        num_objects = torch.randint(1, 4, (1,)).item() # 배치당 1-3개의 객체
        for _ in range(num_objects):
            # 객체 중심 (공간 범위 내)
            center_d = torch.randint(0, spatial_shape[0], (1,)).item()
            center_h = torch.randint(0, spatial_shape[1], (1,)).item()
            center_w = torch.randint(0, spatial_shape[2], (1,)).item()

            # 객체 크기
            obj_size = torch.randint(2, 5, (1,)).item() # 2x2x2 에서 4x4x4 크기

            # 객체 포인트 생성
            obj_coords_d = torch.arange(max(0, center_d - obj_size // 2), min(spatial_shape[0], center_d + obj_size // 2 + 1), device=device)
            obj_coords_h = torch.arange(max(0, center_h - obj_size // 2), min(spatial_shape[1], center_h + obj_size // 2 + 1), device=device)
            obj_coords_w = torch.arange(max(0, center_w - obj_size // 2), min(spatial_shape[2], center_w + obj_size // 2 + 1), device=device)

            # 모든 조합 생성
            coords_grid = torch.stack(torch.meshgrid(obj_coords_d, obj_coords_h, obj_coords_w, indexing='ij'), dim=-1).reshape(-1, 3)

            # 강도 할당 (객체마다 다름)
            intensity = (torch.rand(1, device=device) * 0.5 + 0.5).item() # 0.5 ~ 1.0
            intensities = torch.full((coords_grid.shape[0], 1), intensity, device=device)

            if coords_grid.shape[0] > 0:
                scene_points.append(coords_grid.float())
                scene_intensities.append(intensities)

        # 2. 배경 노이즈 생성
        num_noise_voxels = target_num_voxels - sum([p.shape[0] for p in scene_points])
        num_noise_voxels = max(0, num_noise_voxels) # 최소 0개

        if num_noise_voxels > 0:
            noise_coords = torch.empty((num_noise_voxels, 3), device=device, dtype=torch.float32)
            noise_coords[:, 0] = torch.randint(0, spatial_shape[0], (num_noise_voxels,), device=device).float()
            noise_coords[:, 1] = torch.randint(0, spatial_shape[1], (num_noise_voxels,), device=device).float()
            noise_coords[:, 2] = torch.randint(0, spatial_shape[2], (num_noise_voxels,), device=device).float()

            noise_intensities = torch.rand(num_noise_voxels, 1, device=device) * 0.3 # 0.0 ~ 0.3 (낮은 강도)

            scene_points.append(noise_coords)
            scene_intensities.append(noise_intensities)

        if not scene_points: # 객체도 노이즈도 없는 경우
            # 최소한의 노이즈라도 생성
            coords = torch.empty((1, 3), device=device, dtype=torch.float32)
            coords[:, 0] = torch.randint(0, spatial_shape[0], (1,), device=device).float()
            coords[:, 1] = torch.randint(0, spatial_shape[1], (1,), device=device).float()
            coords[:, 2] = torch.randint(0, spatial_shape[2], (1,), device=device).float()
            feats = torch.cat([coords, torch.rand(1, 1, device=device) * 0.3], dim=1)
            batch_coords = torch.cat([torch.full((1, 1), b, device=device, dtype=torch.int32), coords.int()], dim=1)
            all_coords.append(batch_coords)
            all_feats.append(feats)
            continue


        # 3. 모든 포인트 결합 및 중복 제거 (복셀화)
        # float 좌표와 강도를 결합하여 함께 처리합니다.
        combined_points = torch.cat(scene_points, dim=0)
        combined_intensities = torch.cat(scene_intensities, dim=0)

        # 복셀 좌표로 반올림
        voxel_coords_float = torch.round(combined_points).long()
        # 경계 내에 있는지 확인
        voxel_coords_float[:, 0].clamp_(0, spatial_shape[0] - 1)
        voxel_coords_float[:, 1].clamp_(0, spatial_shape[1] - 1)
        voxel_coords_float[:, 2].clamp_(0, spatial_shape[2] - 1)

        # 고유한 복셀을 찾고 강도를 평균화 (또는 다른 집계)
        # 각 고유 복셀에 대한 강도 값을 저장할 맵
        # (d, h, w) -> (sum_intensity, count, sum_x, sum_y, sum_z)
        unique_voxel_map = {}
        for i in range(voxel_coords_float.shape[0]):
            coord_tuple = tuple(voxel_coords_float[i].tolist())
            intensity_val = combined_intensities[i].item()
            feat_x, feat_y, feat_z = combined_points[i, 0].item(), combined_points[i, 1].item(), combined_points[i, 2].item()
            
            if coord_tuple not in unique_voxel_map:
                unique_voxel_map[coord_tuple] = [intensity_val, 1, feat_x, feat_y, feat_z]
            else:
                unique_voxel_map[coord_tuple][0] += intensity_val
                unique_voxel_map[coord_tuple][1] += 1
                unique_voxel_map[coord_tuple][2] += feat_x
                unique_voxel_map[coord_tuple][3] += feat_y
                unique_voxel_map[coord_tuple][4] += feat_z


        # 고유한 복셀 좌표와 평균 강도로 텐서 재구성
        unique_coords_list = []
        unique_feats_list = []

        for coord_tuple, data in unique_voxel_map.items():
            sum_intensity, count, sum_x, sum_y, sum_z = data
            avg_intensity = sum_intensity / count
            avg_x, avg_y, avg_z = sum_x / count, sum_y / count, sum_z / count
            
            unique_coords_list.append(torch.tensor(coord_tuple, device=device, dtype=torch.int32))
            # 현재는 (x, y, z, intensity)이므로 x, y, z를 특징으로 포함
            # 실제로는 이 특징이 raw 포인트 클라우드의 특징이 됩니다.
            if in_channels == 4:
                # x, y, z 좌표와 평균 강도를 특징으로 사용
                unique_feats_list.append(torch.cat([
                    torch.tensor([avg_x, avg_y, avg_z], device=device, dtype=torch.float32),
                    torch.tensor([avg_intensity], device=device, dtype=torch.float32)
                ]))
            else: # 다른 in_channels 시나리오를 위한 더미 특징
                unique_feats_list.append(torch.randn(in_channels, device=device, dtype=torch.float32))


        if not unique_coords_list: # 여전히 비어있는 경우 (발생할 가능성 낮음)
            # 최소한의 노이즈라도 생성
            coords = torch.empty((1, 3), device=device, dtype=torch.int32)
            coords[:, 0] = torch.randint(0, spatial_shape[0], (1,), device=device).int()
            coords[:, 1] = torch.randint(0, spatial_shape[1], (1,), device=device).int()
            coords[:, 2] = torch.randint(0, spatial_shape[2], (1,), device=device).int()
            feats = torch.cat([coords.float(), torch.rand(1, 1, device=device) * 0.3], dim=1) if in_channels == 4 else torch.randn(1, in_channels, device=device)
            batch_coords = torch.cat([torch.full((1, 1), b, device=device, dtype=torch.int32), coords], dim=1)
            all_coords.append(batch_coords)
            all_feats.append(feats)
            continue

        batch_coords = torch.stack(unique_coords_list, dim=0)
        batch_feats = torch.stack(unique_feats_list, dim=0)

        # 배치 인덱스 추가
        batch_coords = torch.cat([torch.full((batch_coords.shape[0], 1), b, device=device, dtype=torch.int32), batch_coords], dim=1)
        all_coords.append(batch_coords)
        all_feats.append(batch_feats)

    coords = torch.cat(all_coords, dim=0).int()
    feats = torch.cat(all_feats, dim=0).float()

    return SparseTensor(feats=feats, coords=coords, spatial_shape=spatial_shape, batch_size=batch_size)


class SECONDLikeModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # SECOND 모델의 희소 3D 인코더 부분을 모방합니다.
        # 실제 SECOND는 더 많은 레이어와 복잡한 구조를 가집니다.
        # SparseTriton의 Conv3d 모듈은 희소 컨볼루션을 수행합니다.

        # Layer 1: 입력 특징 채널을 늘리고 공간 해상도를 줄입니다.
        self.conv3d_1 = Conv3d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, subm=False
        )
        self.norm_1 = SparseBatchNorm(32) # nn.BatchNorm1d 대신 SparseBatchNorm 사용
        self.relu_1 = ReLU() # nn.ReLU 대신 SparseTriton의 ReLU 사용

        # Layer 2: 더 깊은 특징을 학습합니다.
        self.conv3d_2 = Conv3d(
            32, 64, kernel_size=3, stride=1, padding=1, subm=True
        )
        self.norm_2 = SparseBatchNorm(64)
        self.relu_2 = ReLU()

        # Layer 3: 추가적인 공간 다운샘플링
        self.conv3d_3 = Conv3d(
            64, 128, kernel_size=3, stride=2, padding=1, subm=False
        )
        self.norm_3 = SparseBatchNorm(128)
        self.relu_3 = ReLU()

        # --- 이 부분부터는 SECOND 모델의 2D 헤드와 유사한 개념을 가정합니다. ---
        # 실제 SECOND는 3D 특징 맵을 2D Bird's Eye View (BEV) 맵으로 변환합니다.
        # 여기서는 단순히 SparseTensor의 특징(feats)을 사용하여 감지 헤드를 구성합니다.
        # 이는 매우 단순화된 형태이며, 실제 SECOND의 복잡한 2D BEV 변환 및 감지 헤드를 대체하지 않습니다.

        # 간단한 분류 헤드 (이 예제에서는 최종 출력 특징을 기준으로 클래스를 예측)
        self.cls_head = nn.Linear(128, num_classes)
        # 간단한 바운딩 박스 회귀 헤드 (이 예제에서는 최종 출력 특징을 기준으로 박스 파라미터를 예측)
        self.reg_head = nn.Linear(128, 7) # 예를 들어, (x, y, z, dx, dy, dz, heading)

    def forward(self, x: SparseTensor):
        # 희소 3D 인코더 부분
        x = self.conv3d_1(x)
        x = self.norm_1(x) # SparseBatchNorm은 SparseTensor를 직접 받습니다.
        x = self.relu_1(x) # SparseTriton의 ReLU도 SparseTensor를 직접 받습니다.

        x = self.conv3d_2(x)
        x = self.norm_2(x)
        x = self.relu_2(x)

        x = self.conv3d_3(x)
        x = self.norm_3(x)
        x = self.relu_3(x)

        # 최종 출력 특징 (각 활성화된 복셀의 특징)
        final_features = x.F # 이 특징들을 사용하여 감지 헤드를 구성합니다.

        # 2D 헤드 (매우 단순화됨)
        # 실제 SECOND에서는 이 final_features를 2D BEV 맵으로 변환한 후 2D CNN을 통과시킵니다.
        # 여기서는 모든 특징에 대해 직접 선형 레이어를 적용합니다.
        logits = self.cls_head(final_features)
        bboxes_pred = self.reg_head(final_features)

        # 실제 학습에서는 logits와 bboxes_pred를 사용하여 손실을 계산합니다.
        # 여기서는 예시를 위해 최종 출력을 반환합니다.
        return logits, bboxes_pred


if __name__ == "__main__":
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 파라미터
    batch_size = 1 # Oakland 3D 데이터셋은 현재 단일 파일 로드에 초점
    spatial_shape = (48, 256, 256)  # 예시: 48 깊이, 256x256 해상도
    in_channels = 4               # 예시: (x, y, z, intensity)
    num_classes = 3               # 예시: Car, Pedestrian, Cyclist

    # Oakland 3D 데이터셋 파라미터
    # 실제 데이터셋의 범위에 맞게 조정해야 합니다.
    # Oakland 3D 데이터셋 문서를 참조하여 적절한 값을 설정하세요.
    # 여기서는 임의의 값을 사용합니다.
    voxel_size = (0.4, 0.2, 0.2) # (z, y, x) 순서로 대략적인 복셀 크기 (미터)
    # point_cloud_range: [min_x, max_x, min_y, max_y, min_z, max_z]
    point_cloud_range = [-50.0, 50.0, -50.0, 50.0, -5.0, 5.0]

    # Oakland 3D 데이터 파일 경로 (사용자 제공)
    # 여기에 실제 파일 경로를 지정해야 합니다.
    oakland_data_dir = "examples/Oakland3D/training_data/training"
    oakland_file_name = "oakland_part3_an_training.xyz_label_conf"
    full_oakland_file_path = os.path.join(oakland_data_dir, oakland_file_name)

    # 모델 인스턴스 생성
    model = SECONDLikeModel(in_channels, num_classes).to(device)
    print("SECONDLikeModel 구조:")
    print(model)

    # 실제 희소 입력 데이터 로드
    print(f"\nOakland 3D 데이터셋 로드: {full_oakland_file_path}")
    real_input = load_oakland_sparse_data(
        file_path=full_oakland_file_path,
        spatial_shape=spatial_shape,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        in_channels=in_channels,
        batch_idx=0 # 단일 파일이므로 배치 인덱스 0
    )
    print(f"\n입력 SparseTensor: {real_input}")
    print(f"  - 특징 텐서 shape: {real_input.F.shape}")
    print(f"  - 좌표 텐서 shape: {real_input.C.shape}")
    print(f"  - 공간 shape: {real_input.spatial_shape}")
    print(f"  - 배치 크기: {real_input.batch_size}")

    # 모델 순전파
    print("\n모델 순전파 시작...")
    # SparseTensor가 비어있지 않은 경우에만 순전파를 시도합니다.
    if real_input.F.shape[0] > 0:
        logits, bboxes_pred = model(real_input)
        print("모델 순전파 완료.")

        # 출력 결과 확인
        print(f"\n예측된 클래스 로짓 shape: {logits.shape}")
        print(f"예측된 바운딩 박스 shape: {bboxes_pred.shape}")
    else:
        print("입력 SparseTensor가 비어 있어 모델 순전파를 건너뜀.")

    # 참고: 실제 SECOND 모델 학습에는 데이터 로딩, 손실 함수, 옵티마이저,
    # 복잡한 2D BEV 처리, 타겟 생성 등 훨씬 더 많은 구성 요소가 필요합니다.
    # 이 예제는 SparseTriton의 Conv3d 사용법과 실제 데이터셋 통합을 보여주는 데 중점을 둡니다.
