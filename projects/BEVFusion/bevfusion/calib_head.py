import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet3d.registry import MODELS

def axis_angle_to_matrix(axis_angle: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    하나 이상의 축-각 벡터를 3x3 회전 행렬로 변환합니다. (ver 3.0, 최종 안정화)
    
    0-벡터(zero-angle) 입력에 대해 nan 그래디언트가 발생하지 않도록
    각도(theta)가 epsilon보다 작은 경우와 큰 경우를 torch.where로 완벽히 분리하여
    불안정한 그래디언트 계산 경로 자체를 차단합니다.
    
    Args:
        axis_angle (torch.Tensor): 변환할 축-각 벡터. Shape: (..., 3)
        epsilon (float): 0으로 간주할 매우 작은 각도(의 제곱) 기준값.
                        dtype에 따라 (예: float16) 1e-6 정도로 높여야 할 수 있습니다.
    Returns:
        torch.Tensor: 변환된 회전 행렬. Shape: (..., 3, 3)
    """
    device = axis_angle.device
    dtype = axis_angle.dtype

    # ✨ FIX: 입력 자체의 NaN/Inf 방어
    # 상위 네트워크에서 Inf/NaN이 터져서 넘어올 경우를 대비한 방어 코드
    axis_angle = torch.nan_to_num(axis_angle, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. 각도(theta)의 제곱(theta^2) 계산
    # (B, 1) 또는 (..., 1)
    angle_sq = torch.sum(axis_angle**2, dim=-1, keepdim=True)

    # 2. 정규화되지 *않은* 축-각 벡터 v로 Skew-symmetric 행렬 K_v 생성
    # (B, 3, 3) 또는 (..., 3, 3)
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, device=device, dtype=dtype)
    K[..., 0, 1] = -axis_angle[..., 2]
    K[..., 0, 2] =  axis_angle[..., 1]
    K[..., 1, 0] =  axis_angle[..., 2]
    K[..., 1, 2] = -axis_angle[..., 0]
    K[..., 2, 0] = -axis_angle[..., 1]
    K[..., 2, 1] =  axis_angle[..., 0]

    # 단위 행렬 I 생성
    I = torch.eye(3, device=device, dtype=dtype).expand_as(K)
    
    # K_v의 제곱 계산
    K_sq = torch.matmul(K, K)

    # 3. 로드리게스 공식을 위한 계수 A, B 계산
    
    # (B, 1) 또는 (..., 1)
    small_angle_mask = (angle_sq < epsilon)
    
    # --- Case 1: 각도가 매우 작은 경우 (Taylor Expansion) ---
    # A ≈ 1 - θ²/6,  B ≈ 1/2 - θ²/24
    # 이 경로는 0으로 나누는 연산이 아예 없으므로 항상 안전합니다.
    A_small = 1.0 - angle_sq / 6.0
    B_small = 0.5 - angle_sq / 24.0
    
    # --- Case 2: 각도가 0에 가깝지 않은 경우 (일반 계산) ---
    # ✨ FIX: small_angle_mask가 True인 곳은 0이 아닌 1.0으로 대체하여
    # 0으로 나누는 연산 자체를 방지합니다.
    # (어차피 이 값들은 small_angle_mask에 의해 버려짐)
    angle_sq_safe = torch.where(small_angle_mask, 
                              torch.ones_like(angle_sq), 
                              angle_sq)
    
    angle_safe = torch.sqrt(angle_sq_safe)
    
    sin_angle = torch.sin(angle_safe)
    cos_angle = torch.cos(angle_safe)
    
    A_large = sin_angle / angle_safe
    B_large = (1.0 - cos_angle) / angle_sq_safe
    
    # --- 두 케이스 병합 ---
    # [..., 1, 1] 로 브로드캐스팅 준비
    # small_angle_mask가 True인 곳은 A_small/B_small (안전한 경로)
    # False인 곳은 A_large/B_large (0이 아님이 보장된 경로)
    A = torch.where(small_angle_mask, A_small, A_large).unsqueeze(-1)
    B = torch.where(small_angle_mask, B_small, B_large).unsqueeze(-1)
    
    # 4. 로드리게스 회전 공식 적용
    # R = I + A * K_v + B * K_v²
    rotation_matrix = I + A * K + B * K_sq
    
    return rotation_matrix

def geodesic_distance_loss(R_pred, R_gt, epsilon=1e-7):
    """두 회전 행렬 간의 측지 거리(각도 차이) 계산"""
    R_rel = torch.matmul(R_pred, R_gt.transpose(-2, -1))
    trace = torch.diagonal(R_rel, offset=0, dim1=-2, dim2=-1).sum(-1)
    trace = torch.clamp(trace, -1.0 + epsilon, 3.0 - epsilon)
    angle = torch.acos((trace - 1) / 2.0)
    return angle

# --- 새로 추가된 헬퍼 함수 ---
def create_transformation_matrix(rot_vec, trans_vec):
    """3D 회전 벡터와 3D 이동 벡터로부터 4x4 동차 변환 행렬 생성"""
    batch_size = rot_vec.shape[0]
    rotation_matrix = axis_angle_to_matrix(rot_vec) # [B, 3, 3]
    
    # 4x4 행렬 생성 (기본은 단위 행렬 형태)
    transformation_matrix = torch.eye(4, device=rot_vec.device, dtype=rot_vec.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 회전 부분 채우기
    transformation_matrix[:, :3, :3] = rotation_matrix
    
    # 이동 부분 채우기
    transformation_matrix[:, :3, 3] = trans_vec
    
    return transformation_matrix

# --- 카메라 제안 보정 함수 (핵심 로직) ---
def correct_camera_proposals(det_xyz_norm, pred_delta_rot, pred_delta_trans, pc_range_tensor):
    """
    예측된 캘리브레이션 오차로 정규화된 카메라 제안 좌표를 보정합니다.
    Args:
        det_xyz_norm (Tensor): [B, NumProposals, 3] 정규화된 좌표 [0,1]
        pred_delta_rot (Tensor): [B, 3] 예측된 회전 오차 벡터
        pred_delta_trans (Tensor): [B, 3] 예측된 이동 오차 벡터
        pc_range_tensor (Tensor): 포인트 클라우드 범위 [xmin, ymin, zmin, xmax, ymax, zmax]
    Returns:
        Tensor: [B, NumProposals, 3] 보정 후 다시 정규화된 좌표 [0,1]
    """
    B, N, _ = det_xyz_norm.shape
    device = det_xyz_norm.device

    # 1. 역정규화: [0,1] -> 실제 미터 좌표
    pc_min = pc_range_tensor[:3]
    pc_max = pc_range_tensor[3:]
    pc_range_dims = pc_max - pc_min
    
    det_xyz_metric = det_xyz_norm * pc_range_dims.unsqueeze(0).unsqueeze(1) + pc_min.unsqueeze(0).unsqueeze(1)

    # 2. 보정 변환 행렬 생성
    #    pred_delta_*는 각 배치 샘플에 대해 하나의 값만 가지므로, 모든 제안에 동일하게 적용
    correction_matrix = create_transformation_matrix(pred_delta_rot, pred_delta_trans) # [B, 4, 4]

    # 3. 보정 변환 적용
    points_h = F.pad(det_xyz_metric, (0, 1), mode='constant', value=1.0) # [B, N, 4]
    # 행렬 곱셈을 위해 차원 조정 및 반복:
    # correction_matrix: [B, 4, 4] -> [B, 1, 4, 4] -> [B, N, 4, 4]
    # points_h: [B, N, 4] -> [B, N, 4, 1]
    points_corrected_h = (correction_matrix.unsqueeze(1).expand(-1, N, -1, -1) @ points_h.unsqueeze(-1)).squeeze(-1)
    det_xyz_corrected_metric = points_corrected_h[..., :3]

    # 4. 다시 정규화: 실제 미터 좌표 -> [0,1]
    det_xyz_corrected_norm = (det_xyz_corrected_metric - pc_min.unsqueeze(0).unsqueeze(1)) / pc_range_dims.unsqueeze(0).unsqueeze(1)
    det_xyz_corrected_norm = det_xyz_corrected_norm.clamp(min=0, max=1)

    return det_xyz_corrected_norm

@MODELS.register_module()
class CalibrationCorrectionHead(nn.Module):
    """
    특징 맵을 입력받아 6-DoF 보정 파라미터를 예측하는 헤드.

    Args:
        in_channels (int): 입력 특징 맵의 채널 수.
        hidden_dim (int): MLP의 중간층 차원.
        out_dim (int): 출력 차원. 기본값은 6 (rot 3 + trans 3).
    """
    def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 6):
        super().__init__()
        
        # 1. 공간 차원(H, W)을 없애고 채널 정보만 남기기 위한 풀링 레이어
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. 풀링된 특징 벡터를 최종 6-DoF 값으로 매핑하는 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 특징 맵 (B*N, C, H, W)
        
        Returns:
            torch.Tensor: 예측된 6-DoF 파라미터 (B*N, 6)
        """
        # (B*N, C, H, W) -> (B*N, C, 1, 1)
        x = self.pool(x)
        
        # (B*N, C, 1, 1) -> (B*N, C)
        x = torch.flatten(x, 1)
        
        # (B*N, C) -> (B*N, 6)
        pred_delta_6dof = self.mlp(x)
        
        return pred_delta_6dof