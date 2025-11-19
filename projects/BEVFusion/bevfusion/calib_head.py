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

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    (..., 4) 모양의 쿼터니언을 (..., 3, 3) 회전 행렬로 변환합니다.
    (F.normalize 안정화 버전)
    
    쿼터니언 순서는 (w, x, y, z)로 가정합니다.
    """
    # ✨✨✨ START: 여기가 핵심 수정 ✨✨✨
    # 0-벡터(Zero Vector)가 입력될 경우 NaN이 발생하는 것을 막기 위해
    # 분모에 작은 값(epsilon)을 더해줍니다.
    norm = torch.norm(quaternions, p=2, dim=-1, keepdim=True)
    eps = 1e-8 # 0으로 나누기 방지
    quaternions_normalized = quaternions / (norm + eps)
    # ✨✨✨ END: 여기가 핵심 수정 ✨✨✨

    # 정규화된 쿼터니언 사용
    w, x, y, z = quaternions_normalized[..., 0], quaternions_normalized[..., 1], \
                   quaternions_normalized[..., 2], quaternions_normalized[..., 3]

    # 공통 계산 항목
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # 회전 행렬 계산 (이 공식은 NaN을 유발하는 나눗셈이 없음)
    R = torch.empty(*quaternions.shape[:-1], 3, 3, device=quaternions.device, dtype=quaternions.dtype)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    
    return R

def identity_matrix_loss(R_pred, R_gt):
    """
    R_pred와 R_gt의 상대 회전 행렬 R_rel이 
    단위 행렬(Identity) I에 얼마나 가까운지 MSE로 측정합니다.
    (acos를 사용하지 않아 매우 안정적입니다.)
    """
    # 1. 상대 회전 행렬 계산 (이전과 동일)
    R_rel = torch.matmul(R_pred, R_gt.transpose(-2, -1))
    
    # 2. 타겟이 될 단위 행렬(Identity) 생성
    # R_rel과 동일한 shape, device, dtype을 갖는 단위 행렬 I를 만듭니다.
    I = torch.eye(3, device=R_rel.device, dtype=R_rel.dtype)
    # 배치 크기(B*N)만큼 I를 확장합니다.
    I = I.expand_as(R_rel) # (B*N, 3, 3)

    # 3. R_rel과 I의 MSE Loss 계산
    # R_rel이 I와 완벽히 같다면 이 Loss는 0이 됩니다.
    loss = F.mse_loss(R_rel, I, reduction='mean')
    
    return loss

def geodesic_distance_loss(R_pred, R_gt, epsilon=1e-7):
    """두 회전 행렬 간의 측지 거리(각도 차이) 계산"""
    R_rel = torch.matmul(R_pred, R_gt.transpose(-2, -1))
    trace = torch.diagonal(R_rel, offset=0, dim1=-2, dim2=-1).sum(-1)
    trace = torch.clamp(trace, -1.0 + epsilon, 3.0 - epsilon)
    angle = torch.acos((trace - 1) / 2.0)
    return angle

# # --- 새로 추가된 헬퍼 함수 ---
# def create_transformation_matrix(rot_vec, trans_vec):
#     """3D 회전 벡터와 3D 이동 벡터로부터 4x4 동차 변환 행렬 생성"""
#     batch_size = rot_vec.shape[0]
#     rotation_matrix = axis_angle_to_matrix(rot_vec) # [B, 3, 3]
    
#     # 4x4 행렬 생성 (기본은 단위 행렬 형태)
#     transformation_matrix = torch.eye(4, device=rot_vec.device, dtype=rot_vec.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    
#     # 회전 부분 채우기
#     transformation_matrix[:, :3, :3] = rotation_matrix
    
#     # 이동 부분 채우기
#     transformation_matrix[:, :3, 3] = trans_vec
    
#     return transformation_matrix

# --- ✨ START: 수정할 함수 ✨ ---
def create_transformation_matrix(rot_input: torch.Tensor, trans_vec: torch.Tensor) -> torch.Tensor:
    """
    3D 회전 벡터(3D) 또는 쿼터니언(4D)과 3D 이동 벡터로부터 
    4x4 동차 변환 행렬을 생성합니다.

    Args:
        rot_input (torch.Tensor): (..., 3) [축-각] 또는 (..., 4) [쿼터니언]
        trans_vec (torch.Tensor): (..., 3) [이동 벡터]
    
    Returns:
        torch.Tensor: (..., 4, 4) 변환 행렬
    """
    batch_shape = rot_input.shape[:-1]
    
    if rot_input.shape[-1] == 3:
        # 입력이 3D (축-각)인 경우
        rotation_matrix = axis_angle_to_matrix(rot_input) # (..., 3, 3)
    elif rot_input.shape[-1] == 4:
        # 입력이 4D (쿼터니언)인 경우
        rotation_matrix = quaternion_to_matrix(rot_input) # (..., 3, 3)
    else:
        raise ValueError(
            f"Unknown rotation format. Expected 3 (axis-angle) or 4 (quaternion) "
            f"dims in the last axis, got {rot_input.shape[-1]}"
        )

    # 4x4 행렬 생성 (기본은 단위 행렬 형태)
    # .expand() 대신 torch.eye.repeat()를 사용하여 배치 차원에 맞게 생성
    transformation_matrix = torch.eye(
        4, device=rot_input.device, dtype=rot_input.dtype
    ).unsqueeze(0).repeat(*batch_shape, 1, 1) # (..., 4, 4)
    
    # 회전 부분 채우기
    transformation_matrix[..., :3, :3] = rotation_matrix
    
    # 이동 부분 채우기
    transformation_matrix[..., :3, 3] = trans_vec
    
    return transformation_matrix
# --- ✨ END: 수정할 함수 ✨ ---

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

# @MODELS.register_module()
# class CalibrationCorrectionHead(nn.Module):
#     """
#     특징 맵을 입력받아 6-DoF 보정 파라미터를 예측하는 헤드.

#     Args:
#         in_channels (int): 입력 특징 맵의 채널 수.
#         hidden_dim (int): MLP의 중간층 차원.
#         out_dim (int): 출력 차원. 기본값은 6 (rot 3 + trans 3).
#     """
#     def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 6):
#         super().__init__()
        
#         # 1. 공간 차원(H, W)을 없애고 채널 정보만 남기기 위한 풀링 레이어
#         self.pool = nn.AdaptiveAvgPool2d(1)
        
#         # 2. 풀링된 특징 벡터를 최종 6-DoF 값으로 매핑하는 MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): 입력 특징 맵 (B*N, C, H, W)
        
#         Returns:
#             torch.Tensor: 예측된 6-DoF 파라미터 (B*N, 6)
#         """
#         # (B*N, C, H, W) -> (B*N, C, 1, 1)
#         x = self.pool(x)
        
#         # (B*N, C, 1, 1) -> (B*N, C)
#         x = torch.flatten(x, 1)
        
#         # (B*N, C) -> (B*N, 6)
#         pred_delta_6dof = self.mlp(x)
        
#         return pred_delta_6dof
    
# --------------------------------------------------------------------
# 1. DepthCalibTranformer의 'regressor' 로직을 위한 헬퍼 클래스
#    (CalibrationCorrectionHead 클래스보다 *먼저* 정의되어야 합니다)
# --------------------------------------------------------------------
class _CalibHeadRegressor(nn.Module):
    """
    DepthCalibTranformer의 regressor 로직을 구현한 내부 헬퍼 클래스.
    BEVFusion의 2D 입력(num_kp*6)과 쿼터니언(4D) 출력에 맞게 수정됨.
    """
    def __init__(self, in_channels=312, dropout=0.5, num_kp=200):
        super(_CalibHeadRegressor, self).__init__()
        self.num_kp = num_kp
        self.mish = nn.Mish()
        self.dropout2 = nn.Dropout(dropout)
        
        # 1. Global Feature (enc_out) 처리용
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # --- ✨ 1. Local Feature 차원 수정 (6 -> 8) ---
        # (u,v), (u',v',z'), (u-u'), (v-v'), (z') = 8 dims per Kp
        self.corrs_emb_dim = self.num_kp * 7
        
        # 3. MLP 입력 차원 = Global (in_channels) + Local (corrs_emb_dim)
        self.mlp_input_dim = self.corrs_emb_dim + in_channels # 예: (200 * 6) + 312 = 1512

        # 4. 회전(Rotation) 브랜치 (regressor와 동일 구조)
        self.fc0_rot_aggr = nn.Linear(self.mlp_input_dim, 1024)
        self.bn0_rot_aggr = nn.BatchNorm1d(1024)
        self.fc0_rot = nn.Linear(1024, 512)
        self.bn0_rot = nn.BatchNorm1d(512)
        self.fc1_rot = nn.Linear(512, 256)
        self.bn1_rot = nn.BatchNorm1d(256)
        # --- ✨ 수정: NaN 방지를 위해 쿼터니언(4D) 출력 ---
        self.fc2_rot = nn.Linear(256, 3) 

        # 5. 이동(Translation) 브랜치 (regressor와 동일 구조)
        self.fc0_tarsl_aggr = nn.Linear(self.mlp_input_dim, 1024)
        self.bn0_tarsl_aggr = nn.BatchNorm1d(1024)
        self.fc0_trasl = nn.Linear(1024, 512)
        self.bn0_trasl = nn.BatchNorm1d(512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.bn1_trasl = nn.BatchNorm1d(256)
        self.fc2_trasl = nn.Linear(256, 3) # 3D translation

    def forward(self, x_global, y_local_flat):
        """
        Args:
            x_global (Tensor): (B*N, C) - Global AvgPool Feature
            y_local_flat (Tensor): (B*N, num_kp*6) - Flattened Corrs Feature
        """
        # (B*N, C + num_kp*6)
        feature_emb = torch.cat((x_global, y_local_flat), dim=-1)
        
        # --- 회전 브랜치 ---
        aggr_rot_x = self.mish(self.bn0_rot_aggr(self.fc0_rot_aggr(feature_emb)))
        aggr_rot_x = self.dropout2(aggr_rot_x)
        rot = self.mish(self.bn0_rot(self.fc0_rot(aggr_rot_x)))
        rot = self.mish(self.bn1_rot(self.fc1_rot(rot)))
        rot = self.fc2_rot(rot) # (B*N, 3) 축-각 원복
        # rot = torch.tanh(self.fc2_rot(rot))

        # --- 이동 브랜치 ---
        aggr_transl_x = self.mish(self.bn0_tarsl_aggr(self.fc0_tarsl_aggr(feature_emb)))
        aggr_transl_x = self.dropout2(aggr_transl_x)
        transl = self.mish(self.bn0_trasl(self.fc0_trasl(aggr_transl_x)))
        transl = self.mish(self.bn1_trasl(self.fc1_trasl(transl)))
        transl = self.fc2_trasl(transl) # (B*N, 3)

        return rot, transl

# --------------------------------------------------------------------
# 2. 새로운 CalibrationCorrectionHead (기존 스텁 덮어쓰기)
# --------------------------------------------------------------------
@MODELS.register_module()
class CalibrationCorrectionHead(nn.Module):
    """
    특징 맵(enc_out)과 대응점(corrs)을 입력받아
    'regressor' 로직을 사용해 6-DoF 보정 파라미터를 예측하는 헤드.
    (쿼터니언 4D + 이동 3D = 7D 출력)

    Args:
        in_channels (int): enc_out의 채널 수. (기본값: 312)
        num_kp (int): correspondence keypoint의 수. (기본값: 200)
        dropout_p (float): 드롭아웃 확률. (기본값: 0.5)
    """
    def __init__(self, in_channels: int = 312, num_kp: int = 200, dropout_p: float = 0.5):
        super().__init__()
        self.num_kp = num_kp
        
        # regressor 로직을 포함하는 내부 모듈 생성
        self.regressor = _CalibHeadRegressor(
            in_channels=in_channels, 
            dropout=dropout_p, 
            num_kp=num_kp
        )

    # def forward(self, 
    #             enc_out: torch.Tensor, 
    #             query_input: torch.Tensor, 
    #             corrs_pred_3d: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         enc_out (torch.Tensor): (B*N, C, H, W) e.g., (B*N, 312, 12, 64)
    #         query_input (torch.Tensor): (B*N, N_kp, 2) e.g., (B*N, 200, 2)
    #         corrs_pred (torch.Tensor): (B*N, N_kp, 2) e.g., (B*N, 200, 2)
        
    #     Returns:
    #         torch.Tensor: 예측된 7-DoF 파라미터 (B*N, 7)
    #                      [..., :4] = quaternion (w, x, y, z) 또는 (x, y, z, w)
    #                      [..., 4:] = translation (x, y, z)
    #     """
        
    #     # 1. Global Feature (enc_out) 처리
    #     # (B*N, C, H, W) -> (B*N, C, 1, 1) -> (B*N, C)
    #     x_global = self.regressor.flatten(self.regressor.avgpool(enc_out))
        
    #     # 2. Local Feature (corrs_emb) 처리 (✨ z' 추가 ✨)
        
    #     # (u,v)와 (u',v',z') 결합
    #     # [수정] corrs_pred_3d에서 (u',v')와 (z') 분리
    #     corrs_pred_2d = corrs_pred_3d[..., :2] # (B*N, 200, 2)
    #     corrs_pred_z = corrs_pred_3d[..., 2:3] # (B*N, 200, 1)
        
    #     concat_pred_corrs = torch.cat((query_input, corrs_pred_2d), dim=-1) # (B*N, 200, 4) (u,v, u',v')
        
    #     # (u-u')와 (v-v') 차이 벡터 계산 (동일)
    #     x_diff = concat_pred_corrs[..., 0] - concat_pred_corrs[..., 2] # u - u'
    #     y_diff = concat_pred_corrs[..., 1] - concat_pred_corrs[..., 3] # v - v'
    #     concat_pred_corrs_diff = torch.stack([x_diff, y_diff], dim=2)  # (B*N, 200, 2)
        
    #     # [수정] (u,v, u',v', u-u', v-v') + (z') + (z') = 8D
    #     # (u,v), (u',v',z'), (u-u'), (v-v'), (z') -> 8D
    #     corrs_emb = torch.cat(
    #         (query_input,           # (B*N, 200, 2)
    #          corrs_pred_3d,         # (B*N, 200, 3)
    #          concat_pred_corrs_diff, # (B*N, 200, 2)
    #          corrs_pred_z),          # (B*N, 200, 1) -> z'를 한 번 더 넣어줌 (중요도 강조)
    #         dim=-1
    #     ) # (B*N, 200, 8)
        
    #     # MLP 입력을 위해 (B*N, 200, 8) -> (B*N, 200 * 8)
    #     y_local_flat = corrs_emb.view(corrs_emb.size(0), -1) # (B*N, 1600)
        
    #     # 3. Regressor 호출 (동일)
    #     pred_rot, pred_trans = self.regressor(x_global, y_local_flat)
        
    #     # 4. 결과 결합 (동일)
    #     pred_delta_7dof = torch.cat([pred_rot, pred_trans], dim=1)
        
    #     return pred_delta_7dof
    
    def forward(self, 
                enc_out: torch.Tensor, 
                query_input: torch.Tensor, 
                corrs_pred_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): (B*N, C, H, W) e.g., (B*N, 312, 12, 64)
            query_input (torch.Tensor): (B*N, N_kp, 2) (u: [0, 0.5], v: [0, 1])
            corrs_pred_3d (torch.Tensor): (B*N, N_kp, 3) (u': [0.5, 1], v': [0, 1], z': [0, 80])
        ...
        """
        
        # 1. Global Feature (enc_out) 처리 (동일)
        x_global = self.regressor.flatten(self.regressor.avgpool(enc_out))
        
        # --- 2. Local Feature (corrs_emb) 처리 (✨ [0, 1] 정규화 ✨) ---
        
        # 2a. 입력 텐서 분리
        query_u = query_input[..., 0:1] # (B*N, 200, 1) - [0, 0.5]
        query_v = query_input[..., 1:2] # (B*N, 200, 1) - [0, 1]
        
        corrs_u_prime = corrs_pred_3d[..., 0:1] # (B*N, 200, 1) - [0.5, 1]
        corrs_v_prime = corrs_pred_3d[..., 1:2] # (B*N, 200, 1) - [0, 1]
        corrs_z_prime = corrs_pred_3d[..., 2:3] # (B*N, 200, 1) - [0, 80]

        # 2b. 모든 입력을 [0, 1] 범위로 정규화
        # [0, 0.5]  -> [0, 1]  (x * 2.0)
        query_u_norm = query_u * 2.0
        
        # [0.5, 1]  -> [0, 1]  ( (x - 0.5) * 2.0 )
        corrs_u_prime_norm = (corrs_u_prime - 0.5) * 2.0
        
        # [0, 1]    -> [0, 1]  (변경 없음)
        query_v_norm = query_v
        corrs_v_prime_norm = corrs_v_prime
        
        # [0, 80]   -> [0, 1]  (x / 80.0)
        corrs_z_prime_norm = corrs_z_prime / 80.0

        # 2c. [0, 1] 정규화된 좌표로 차이(diff) 벡터 계산
        # (u - u')와 (v - v') 차이 벡터
        x_diff = query_u_norm - corrs_u_prime_norm # [0, 1] - [0, 1] -> [-1, 1]
        y_diff = query_v_norm - corrs_v_prime_norm # [0, 1] - [0, 1] -> [-1, 1]
        
        # diff 벡터도 [0, 1] 범위로 스케일링
        x_diff_norm = (x_diff + 1.0) / 2.0 # [-1, 1] -> [0, 2] -> [0, 1]
        y_diff_norm = (y_diff + 1.0) / 2.0 # [-1, 1] -> [0, 2] -> [0, 1]
        
        concat_pred_corrs_diff_norm = torch.cat([x_diff_norm, y_diff_norm], dim=-1) # (B*N, 200, 2)
        
        # 2d. [0, 1]로 정규화된 모든 특징 결합
        # (u,v) + (u',v') + (z') + (u-u')_norm + (v-v')_norm -> 7D
        corrs_emb = torch.cat(
            (query_u_norm,              # (B*N, 200, 1)
             query_v_norm,              # (B*N, 200, 1)
             corrs_u_prime_norm,        # (B*N, 200, 1)
             corrs_v_prime_norm,        # (B*N, 200, 1)
             corrs_z_prime_norm,        # (B*N, 200, 1)
             concat_pred_corrs_diff_norm),# (B*N, 200, 2)
            dim=-1
        ) # (B*N, 200, 7)
        
        # MLP 입력을 위해 (B*N, 200, 7) -> (B*N, 200 * 7)
        y_local_flat = corrs_emb.view(corrs_emb.size(0), -1) # (B*N, 1400)
        
        # 3. Regressor 호출 (동일)
        pred_rot, pred_trans = self.regressor(x_global, y_local_flat)
        
        # 4. 결과 결합 (동일)
        pred_delta_6dof = torch.cat([pred_rot, pred_trans], dim=1)
        
        return pred_delta_6dof