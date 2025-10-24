import torch
import torch.nn as nn
from torch.nn import functional as F
from mmdet3d.registry import MODELS

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    하나 이상의 축-각 벡터를 3x3 회전 행렬로 변환합니다.

    Args:
        axis_angle (torch.Tensor): 변환할 축-각 벡터. 
                                  Shape: (..., 3), 여기서 ...는 배치 차원을 의미.

    Returns:
        torch.Tensor: 변환된 회전 행렬. Shape: (..., 3, 3).
    """
    # 1. 각도(theta)와 단위 회전축(axis) 분리
    # 벡터의 크기(norm)가 회전 각도(radian)
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    
    # 수치적 안정성을 위해 작은 epsilon 추가
    # 각도가 0에 가까우면 axis는 어떤 방향이든 상관없음
    axis = F.normalize(axis_angle, dim=-1)

    # 2. 로드리게스 공식을 위한 준비
    # 단위 회전축 벡터로 skew-symmetric cross-product 행렬 K 생성
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] =  axis[..., 1]
    K[..., 1, 0] =  axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] =  axis[..., 0]

    # 단위 행렬 I 생성
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    
    # cos(theta)와 sin(theta) 계산
    # (..., 1) -> (..., 1, 1) 형태로 브로드캐스팅 준비
    cos_angle = torch.cos(angle).unsqueeze(-1)
    sin_angle = torch.sin(angle).unsqueeze(-1)

    # 3. 로드리게스 회전 공식 적용
    # R = I + sin(θ)K + (1 - cos(θ))K^2
    rotation_matrix = I + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
    
    return rotation_matrix

def axis_angle_to_matrix(axis_angle):
    """
    하나 이상의 축-각 회전 벡터 배치를 3x3 회전 행렬 배치로 변환합니다.
    (Rodrigues' formula) - [B, ..., 3] 입력 처리 가능 (안정성 강화 버전)
    Args:
        axis_angle (Tensor): [..., 3] 모양의 회전 벡터.
    Returns:
        Tensor: [..., 3, 3] 모양의 회전 행렬.
    """
    original_shape = axis_angle.shape[:-1] # 마지막 3 제외한 원래 모양 저장
    axis_angle = axis_angle.reshape(-1, 3) # 계산 편의를 위해 [N, 3] 형태로 변경
    num_vectors = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype

    angle = torch.norm(axis_angle, p=2, dim=-1, keepdim=True) # [N, 1]
    
    # 단위 행렬 미리 생성 (기본값)
    eye = torch.eye(3, device=device, dtype=dtype)
    R = eye.unsqueeze(0).repeat(num_vectors, 1, 1) # [N, 3, 3], 초기값은 단위 행렬

    # 각도가 0보다 큰 경우에만 계산 수행 (마스크 생성)
    mask = (angle > 1e-6).squeeze(-1) # [N], True이면 각도가 큼

    if mask.any():
        # 마스크가 True인 벡터들만 선택
        axis_angle_nz = axis_angle[mask] # [num_nz, 3]
        angle_nz = angle[mask]           # [num_nz, 1]
        
        # Normalize axis only for non-zero angles
        axis_nz = F.normalize(axis_angle_nz, p=2, dim=-1) # [num_nz, 3]

        cos_nz = torch.cos(angle_nz) # [num_nz, 1]
        sin_nz = torch.sin(angle_nz) # [num_nz, 1]

        # Skew-symmetric 행렬 생성 [num_nz, 3, 3]
        num_nz = axis_nz.shape[0]
        skew_symmetric_nz = torch.zeros(num_nz, 3, 3, device=device, dtype=dtype)
        skew_symmetric_nz[:, 0, 1] = -axis_nz[:, 2]
        skew_symmetric_nz[:, 0, 2] = axis_nz[:, 1]
        skew_symmetric_nz[:, 1, 0] = axis_nz[:, 2]
        skew_symmetric_nz[:, 1, 2] = -axis_nz[:, 0]
        skew_symmetric_nz[:, 2, 0] = -axis_nz[:, 1]
        skew_symmetric_nz[:, 2, 1] = axis_nz[:, 0]

        # Rodrigues' formula 계산 (브로드캐스팅을 위해 차원 명시적 추가)
        eye_nz = eye.unsqueeze(0).repeat(num_nz, 1, 1) # [num_nz, 3, 3]
        
        # ✨ FIX: sin_nz, (1 - cos_nz)의 차원을 [num_nz, 1, 1]로 명확히 확장 ✨
        term2 = sin_nz.unsqueeze(-1) * skew_symmetric_nz 
        term3_matmul = torch.matmul(skew_symmetric_nz, skew_symmetric_nz)
        term3 = (1 - cos_nz).unsqueeze(-1) * term3_matmul
        
        R_nz = eye_nz + term2 + term3 # [num_nz, 3, 3]
        
        # 계산된 결과를 원래 R 텐서의 해당 위치에 업데이트
        R[mask] = R_nz
        
    # 원래 모양으로 복원 (예: [B, N_cam, 3, 3])
    R = R.view(*original_shape, 3, 3)
    return R

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