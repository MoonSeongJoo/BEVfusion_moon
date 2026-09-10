import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint 
from mmengine import print_log   

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
    T_Error = create_transformation_matrix(pred_delta_rot, pred_delta_trans) # [B, 4, 4]

    # 🛑 [CRITICAL FIX] 2.2 T_Error의 역행렬을 계산하여 T_Correction으로 사용
    try:
        # T_Correction = T_Error_inverse
        T_Correction = torch.linalg.inv(T_Error)
    except torch.linalg.LinAlgError:
        # 역행렬 계산이 불가능한 경우 단위 행렬로 대체 (보정 없음)
        T_Correction = torch.eye(4, dtype=T_Error.dtype, device=T_Error.device).expand_as(T_Error)

    # 3. 보정 변환 적용
    points_h = F.pad(det_xyz_metric, (0, 1), mode='constant', value=1.0) # [B, N, 4]
    # 행렬 곱셈을 위해 차원 조정 및 반복:
    # correction_matrix: [B, 4, 4] -> [B, 1, 4, 4] -> [B, N, 4, 4]
    # points_h: [B, N, 4] -> [B, N, 4, 1]
    points_corrected_h = (T_Correction.unsqueeze(1).expand(-1, N, -1, -1) @ points_h.unsqueeze(-1)).squeeze(-1)
    det_xyz_corrected_metric = points_corrected_h[..., :3]

    # 4. 다시 정규화: 실제 미터 좌표 -> [0,1]
    det_xyz_corrected_norm = (det_xyz_corrected_metric - pc_min.unsqueeze(0).unsqueeze(1)) / pc_range_dims.unsqueeze(0).unsqueeze(1)
    det_xyz_corrected_norm = det_xyz_corrected_norm.clamp(min=0, max=1)

    return det_xyz_corrected_norm

@MODELS.register_module()
class CalibrationCorrectionHead(BaseModule):
    """
    Reliability-Aware Set Aggregation Calibration Head V1.

    Main changes from legacy CalibHead:

    1. NO flattening of Q correspondences.
    2. Same shared Point-MLP is applied to every correspondence.
    3. Rotation / Translation use separate learned reliability weights.
    4. ZEstimator V3 reliability is explicitly provided.
    5. CorrNet local query/correspondence features are explicitly sampled.
    6. LayerNorm is used instead of BatchNorm.
    7. No Transformer.

    Inputs
    ------
    enc_out:
        [M, C, H, W]
        CorrNet encoder feature.

    query_input:
        [M, Q, 2]
        SBS normalized query coordinates.
        u in [0, 0.5], v in [0, 1].

    corrs_pred_3d:
        [M, Q, 3]
        [u', v', z']
        u' in [0.5, 1], v' in [0, 1], z' in meters.

    z_reliability:
        [M, Q, 4]

        channel 0:
            neighborhood valid ratio

        channel 1:
            neighborhood max weight

        channel 2:
            center raw-depth valid indicator

        channel 3:
            normalized disagreement between
            neighborhood anchor and center raw depth

    Returns
    -------
    pred_delta_6dof:
        [M, 6]

    diagnostics:
        dict
    """

    def __init__(
        self,
        in_channels: int = 312,
        num_kp: int = 200,
        dropout_p: float = 0.1,
        local_dim: int = 32,
        point_dim: int = 64,
        global_dim: int = 64,
        # NEW
        rot_condition_dim: int = 16,
        rot_condition_scale_deg: float = 10.0,
        init_cfg=None,
    ):
        super().__init__(
            init_cfg=init_cfg
        )

        self.num_kp = num_kp
        self.in_channels = in_channels


        self.rot_condition_dim = rot_condition_dim
        self.rot_condition_scale_rad = math.radians(
            rot_condition_scale_deg
        )
        self.local_dim = local_dim
        self.point_dim = point_dim
        self.global_dim = global_dim

        self.z_reliability_dim = 4


        # ============================================================
        # 1. CorrNet local feature adaptor
        #
        # 312 -> 32
        #
        # The same compressed feature map is used to sample:
        #
        #   query feature
        #   correspondence feature
        #
        # ============================================================

        self.local_adaptor = nn.Sequential(

            nn.Conv2d(
                in_channels,
                local_dim,
                kernel_size=1,
            ),

            nn.Mish(),
        )


        # ============================================================
        # 2. Global Corr context
        #
        # Keep the useful global context from the old CalibHead,
        # but compress it:
        #
        # 312 -> 64
        #
        # ============================================================

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.global_encoder = nn.Sequential(

            nn.Linear(
                in_channels,
                global_dim,
            ),

            nn.LayerNorm(
                global_dim
            ),

            nn.Mish(),
        )


        # ============================================================
        # 3. Per-point input
        #
        # Geometric / correspondence values:
        #
        #   q_u
        #   q_v
        #   corr_u
        #   corr_v
        #   z
        #   signed_du
        #   signed_dv
        #
        # = 7
        #
        # Z reliability:
        #
        # = 4
        #
        # Local Corr features:
        #
        # query feature       = 32
        # corr feature        = 32
        # abs difference      = 32
        #
        # = 96
        #
        # Total:
        #
        # 7 + 4 + 96 = 107
        #
        # ============================================================

        point_input_dim = (
            7                           # legacy UVZ geometry
            + 3                         # target intrinsic-normalized ray
            + 3                         # source camera XYZ
            + 2                         # ray displacement
            + self.z_reliability_dim    # 4
            + 1                         # adaptive Z gate
            + local_dim * 3             # 96
        )


        # ============================================================
        # 4. Shared per-point encoder
        #
        # SAME MLP is applied to all Q points.
        #
        # This removes slot dependence of the old flatten head.
        #
        # ============================================================

        self.point_encoder = nn.Sequential(

            nn.Linear(
                point_input_dim,
                128,
            ),

            nn.LayerNorm(
                128
            ),

            nn.Mish(),


            nn.Linear(
                128,
                point_dim,
            ),

            nn.LayerNorm(
                point_dim
            ),

            nn.Mish(),
        )


        # ============================================================
        # 5. Separate reliability scoring
        #
        # Rotation and Translation are allowed to trust
        # different correspondences.
        #
        # Translation in particular can learn to care more
        # about Z reliability.
        #
        # ============================================================

        base_score_input_dim = (
            point_dim
            + self.z_reliability_dim
            + 1   # adaptive Z gate
        )

        trans_score_input_dim = (
            base_score_input_dim
            + self.rot_condition_dim
        )

        self.rot_score_head = nn.Sequential(

            nn.Linear(
                base_score_input_dim,
                32,
            ),

            nn.Mish(),

            nn.Linear(
                32,
                1,
            ),
        )

        # ============================================================
        # Rotation -> Translation conditioning
        #
        # Translation should estimate motion after knowing the
        # current rotation estimate.
        #
        # pred_rot:
        #     [M,3]
        #
        # normalized / encoded:
        #     [M,16]
        # ============================================================

        self.rot_condition_encoder = nn.Sequential(

            nn.Linear(
                3,
                self.rot_condition_dim,
                bias=False,
            ),

            nn.LayerNorm(
                self.rot_condition_dim,
            ),

            nn.Mish(),
        )

        self.trans_score_head = nn.Sequential(

            nn.Linear(
                trans_score_input_dim,
                32,
            ),

            nn.Mish(),

            nn.Linear(
                32,
                1,
            ),
        )


        # ============================================================
        # 6. Pose heads
        #
        # Weighted set feature:
        #      64
        #
        # Global Corr context:
        #      64
        #
        # Total:
        #     128
        #
        # ============================================================

        pose_input_dim = (
            point_dim
            + global_dim
        )


        self.rot_head = nn.Sequential(

            nn.Linear(
                pose_input_dim,
                128,
            ),

            nn.LayerNorm(
                128
            ),

            nn.Mish(),

            nn.Dropout(
                dropout_p
            ),


            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(
                64
            ),

            nn.Mish(),


            nn.Linear(
                64,
                3,
            ),
        )

        trans_pose_input_dim = (
            pose_input_dim
            + self.rot_condition_dim
        )
        # 128 + 16 = 144

        self.trans_head = nn.Sequential(

            nn.Linear(
                trans_pose_input_dim,
                128,
            ),

            nn.LayerNorm(
                128
            ),

            nn.Mish(),

            nn.Dropout(
                dropout_p
            ),


            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(
                64
            ),

            nn.Mish(),


            nn.Linear(
                64,
                3,
            ),
        )


        # ============================================================
        # 7. Stable initialization
        #
        # Reliability:
        #
        # Initially all correspondences receive equal weights.
        #
        # Q=200:
        # weight ~= 1 / 200 = 0.005
        #
        # ============================================================

        nn.init.zeros_(
            self.rot_score_head[-1].weight
        )

        nn.init.zeros_(
            self.rot_score_head[-1].bias
        )


        nn.init.zeros_(
            self.trans_score_head[-1].weight
        )

        nn.init.zeros_(
            self.trans_score_head[-1].bias
        )


        # ============================================================
        # Pose starts from identity correction:
        #
        # axis-angle = [0,0,0]
        # translation = [0,0,0]
        #
        # ============================================================

        nn.init.zeros_(
            self.rot_head[-1].weight
        )

        nn.init.zeros_(
            self.rot_head[-1].bias
        )


        nn.init.zeros_(
            self.trans_head[-1].weight
        )

        nn.init.zeros_(
            self.trans_head[-1].bias
        )

        # ============================================================
        # Adaptive Raw-Z vs V3-Z Gate
        #
        # input:
        #   Z reliability                 4
        #   predicted Z / 80              1
        #   raw center Z / 80             1
        #   |pred - raw| / 80             1
        #
        # total = 7
        # ============================================================

        self.depth_gate = nn.Sequential(

            nn.Linear(
                7,
                32,
            ),

            nn.Mish(),

            nn.Linear(
                32,
                1,
            ),
        )

        nn.init.zeros_(
            self.depth_gate[-1].weight
        )

        nn.init.zeros_(
            self.depth_gate[-1].bias
        )


    def _sample_local_feature(
        self,
        feature_map,
        uv_sbs,
    ):
        """
        Args:
            feature_map:
                [M, C_local, H, W]

            uv_sbs:
                [M, Q, 2]
                SBS coordinate in [0,1].

        Returns:
            sampled:
                [M, Q, C_local]
        """

        # [0,1] -> [-1,1]
        grid = (
            uv_sbs
            * 2.0
            - 1.0
        )


        # grid_sample:
        #
        # [M,Q,2]
        # ->
        # [M,1,Q,2]
        grid = (
            grid
            .unsqueeze(1)
        )


        sampled = F.grid_sample(

            feature_map,

            grid,

            mode='bilinear',

            padding_mode='zeros',

            align_corners=False,
        )
        # [M,C,1,Q]


        sampled = (
            sampled
            .squeeze(2)
            .permute(
                0,
                2,
                1,
            )
            .contiguous()
        )
        # [M,Q,C]


        return sampled
    
    # ================================================================
    # NEW: Masked softmax
    #
    # CorrNet still receives Q=200.
    #
    # But duplicated filler queries must receive ZERO weight
    # inside CalibHead aggregation.
    # ================================================================

    def _masked_softmax(
        self,
        logits,
        valid_mask,
    ):

        # logits:
        #     [M, Q, 1]
        #
        # valid_mask:
        #     [M, Q]

        logits = logits.squeeze(-1)

        valid_mask = valid_mask.bool()


        very_negative = (
            torch.finfo(
                logits.dtype
            ).min
        )


        # Invalid/repeated slots cannot participate
        # in softmax competition.
        logits = logits.masked_fill(
            ~valid_mask,
            very_negative,
        )


        weights = F.softmax(
            logits,
            dim=1,
        )


        # Explicitly force repeated slots to zero.
        weights = (
            weights
            * valid_mask.to(
                dtype=weights.dtype
            )
        )


        # Re-normalize over real queries only.
        weights = (
            weights
            /
            weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(
                1e-8
            )
        )


        return weights

    def forward(
        self,
        enc_out: torch.Tensor,
        query_input: torch.Tensor,
        corrs_pred_3d: torch.Tensor,
        z_reliability: torch.Tensor,
        z_raw: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        point_valid_mask: torch.Tensor,
    ):
        """
        CalibrationCorrectionHead V2 forward.

        Main functions
        --------------
        1. Preserve CorrNet fixed Q=200 input.
        2. Exclude duplicated filler correspondences from CalibHead pooling.
        3. Adaptively select/blend Raw-Z and ZEstimator-V3 Z.
        4. Add K-aware geometric representation:
             - target intrinsic-normalized ray
             - source camera XYZ
             - intrinsic-normalized ray displacement
        5. Use separate rotation / translation reliability weights.

        Inputs
        ------
        enc_out:
            [M, C, H, W]

        query_input:
            [M, Q, 2]
            SBS coordinates.
            x in [0, 0.5], y in [0, 1]

        corrs_pred_3d:
            [M, Q, 3]
            [corr_u_sbs, corr_v_sbs, z_pred]

        z_reliability:
            [M, Q, 4]

        z_raw:
            [M, Q, 1]

        camera_intrinsics:
            [M, 3, 3] or [M, 4, 4]

        point_valid_mask:
            [M, Q]

        Returns
        -------
        pred_delta_6dof:
            [M, 6]

        diagnostics:
            dict
        """

        # ============================================================
        # 0. Shape information
        # ============================================================

        M, Q, C3 = corrs_pred_3d.shape


        # ============================================================
        # 1. Runtime guards
        # ============================================================

        if C3 != 3:
            raise RuntimeError(
                '[CalibHead V2] '
                'corrs_pred_3d must be [M,Q,3], '
                f'got {corrs_pred_3d.shape}'
            )


        if enc_out.ndim != 4:
            raise RuntimeError(
                '[CalibHead V2] '
                'enc_out must be [M,C,H,W], '
                f'got {enc_out.shape}'
            )


        if enc_out.shape[0] != M:
            raise RuntimeError(
                '[CalibHead V2] '
                'enc_out batch mismatch: '
                f'enc_out={enc_out.shape}, '
                f'corr={corrs_pred_3d.shape}'
            )


        if (
            query_input.shape[0] != M
            or query_input.shape[1] != Q
            or query_input.shape[-1] != 2
        ):
            raise RuntimeError(
                '[CalibHead V2] '
                'query_input must be [M,Q,2], '
                f'got {query_input.shape}'
            )


        if (
            z_reliability.shape[0] != M
            or z_reliability.shape[1] != Q
            or z_reliability.shape[-1]
            != self.z_reliability_dim
        ):
            raise RuntimeError(
                '[CalibHead V2] '
                'z_reliability must be '
                f'[M,Q,{self.z_reliability_dim}], '
                f'got {z_reliability.shape}'
            )


        if z_raw.ndim == 2:
            z_raw = z_raw.unsqueeze(-1)


        if (
            z_raw.shape[0] != M
            or z_raw.shape[1] != Q
            or z_raw.shape[-1] != 1
        ):
            raise RuntimeError(
                '[CalibHead V2] '
                'z_raw must be [M,Q,1], '
                f'got {z_raw.shape}'
            )


        if camera_intrinsics.ndim != 3:
            raise RuntimeError(
                '[CalibHead V2] '
                'camera_intrinsics must be '
                '[M,3,3] or [M,4,4], '
                f'got {camera_intrinsics.shape}'
            )


        if camera_intrinsics.shape[0] != M:
            raise RuntimeError(
                '[CalibHead V2] '
                'camera intrinsic batch mismatch: '
                f'K={camera_intrinsics.shape}, '
                f'corr={corrs_pred_3d.shape}'
            )


        if (
            camera_intrinsics.shape[-2] < 3
            or camera_intrinsics.shape[-1] < 3
        ):
            raise RuntimeError(
                '[CalibHead V2] '
                'camera_intrinsics must contain '
                'at least a 3x3 intrinsic matrix, '
                f'got {camera_intrinsics.shape}'
            )


        if point_valid_mask.shape != (M, Q):
            raise RuntimeError(
                '[CalibHead V2] '
                'point_valid_mask must be [M,Q], '
                f'got {point_valid_mask.shape}'
            )


        point_valid_mask = point_valid_mask.bool()


        # ============================================================
        # 2. Sanitize reliability
        # ============================================================

        z_reliability = torch.nan_to_num(
            z_reliability,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        z_reliability = z_reliability.clamp(
            min=0.0,
            max=1.0,
        )


        # ============================================================
        # 3. Sanitize SBS coordinates
        #
        # Important:
        # invalid points are masked later, but NaN * 0 is still NaN.
        # Therefore coordinates themselves must be finite here.
        # ============================================================

        query_uv_sbs = torch.nan_to_num(
            query_input[..., :2],
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )


        corr_uv_sbs = torch.nan_to_num(
            corrs_pred_3d[..., :2],
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )


        # ============================================================
        # 4. ZEstimator prediction and raw Z
        # ============================================================

        z_pred = corrs_pred_3d[..., 2:3]


        z_pred = torch.nan_to_num(
            z_pred,
            nan=0.0,
            posinf=80.0,
            neginf=0.0,
        ).clamp(
            min=0.0,
            max=80.0,
        )


        z_raw = torch.nan_to_num(
            z_raw,
            nan=0.0,
            posinf=80.0,
            neginf=0.0,
        ).clamp(
            min=0.0,
            max=80.0,
        )


        # ============================================================
        # 5. Adaptive Raw-Z / Pred-Z Gate
        #
        # gate = 0:
        #     Raw Z
        #
        # gate = 1:
        #     ZEstimator V3 Z
        # ============================================================

        z_gap = (
            torch.abs(
                z_pred - z_raw
            )
            / 80.0
        ).clamp(
            min=0.0,
            max=1.0,
        )


        gate_input = torch.cat(
            [
                z_reliability,      # 4
                z_pred / 80.0,      # 1
                z_raw / 80.0,       # 1
                z_gap,              # 1
            ],
            dim=-1,
        )
        # [M,Q,7]


        # Pure neural-network decision.
        z_gate_learned = torch.sigmoid(
            self.depth_gate(
                gate_input
            )
        )
        # [M,Q,1]


        # Actual working gate.
        z_gate = z_gate_learned


        # ------------------------------------------------------------
        # Current Z reliability convention
        #
        # channel 0:
        #     neighborhood valid ratio
        #
        # channel 2:
        #     raw center-depth valid indicator
        # ------------------------------------------------------------

        center_valid = (
            z_reliability[..., 2:3]
            > 0.5
        )


        neighbor_valid = (
            z_reliability[..., 0:1]
            > 1e-6
        )


        # Neural gate is meaningful only when:
        #
        # - correspondence itself is valid
        # - raw Z exists
        # - neighborhood information exists
        learnable_gate_mask = (
            point_valid_mask
            & center_valid.squeeze(-1)
            & neighbor_valid.squeeze(-1)
        )


        # Raw depth unavailable:
        # force ZEstimator prediction.
        z_gate = torch.where(
            ~center_valid,
            torch.ones_like(z_gate),
            z_gate,
        )


        # Neighborhood unavailable but raw exists:
        # force raw Z.
        z_gate = torch.where(
            center_valid & ~neighbor_valid,
            torch.zeros_like(z_gate),
            z_gate,
        )


        # ============================================================
        # 6. Final Z used by CalibHead
        # ============================================================

        z_used = (
            (1.0 - z_gate) * z_raw
            + z_gate * z_pred
        ).clamp(
            min=0.0,
            max=80.0,
        )


        z_norm = z_used / 80.0


        # ============================================================
        # 7. Legacy UVZ geometry
        #
        # Convert SBS coordinates back to each original camera's
        # normalized image coordinate.
        # ============================================================

        query_u = (
            query_uv_sbs[..., 0:1]
            * 2.0
        )
        # [0,0.5] -> [0,1]


        query_v = (
            query_uv_sbs[..., 1:2]
        )


        corr_u = (
            (
                corr_uv_sbs[..., 0:1]
                - 0.5
            )
            * 2.0
        )
        # [0.5,1] -> [0,1]


        corr_v = (
            corr_uv_sbs[..., 1:2]
        )


        du = query_u - corr_u
        dv = query_v - corr_v


        geom_feature = torch.cat(
            [
                query_u,     # 1
                query_v,     # 1
                corr_u,      # 1
                corr_v,      # 1
                z_norm,      # 1
                du,          # 1
                dv,          # 1
            ],
            dim=-1,
        )
        # [M,Q,7]


        # ============================================================
        # 8. K-aware geometry
        # ============================================================

        K = camera_intrinsics[..., :3, :3]

        K = K.to(
            device=corrs_pred_3d.device,
            dtype=corrs_pred_3d.dtype,
        )


        fx = (
            K[..., 0, 0]
            .reshape(M, 1, 1)
            .clamp_min(1e-6)
        )


        fy = (
            K[..., 1, 1]
            .reshape(M, 1, 1)
            .clamp_min(1e-6)
        )


        cx = (
            K[..., 0, 2]
            .reshape(M, 1, 1)
        )


        cy = (
            K[..., 1, 2]
            .reshape(M, 1, 1)
        )


        # ============================================================
        # 9. Original normalized image coordinates -> pixels
        #
        # nuScenes camera image convention currently used by PCC:
        #
        # W = 1600
        # H = 900
        # ============================================================

        query_u_px = query_u * 1600.0
        query_v_px = query_v * 900.0

        corr_u_px = corr_u * 1600.0
        corr_v_px = corr_v * 900.0


        # ============================================================
        # 10. Intrinsic-normalized target/source coordinates
        #
        # x = (u - cx) / fx = X/Z
        # y = (v - cy) / fy = Y/Z
        # ============================================================

        query_x_cam = (
            query_u_px - cx
        ) / fx


        query_y_cam = (
            query_v_px - cy
        ) / fy


        corr_x_cam = (
            corr_u_px - cx
        ) / fx


        corr_y_cam = (
            corr_v_px - cy
        ) / fy


        # ============================================================
        # 11. Target ray
        #
        # Target query has no independent depth.
        #
        # Keep perspective ray representation:
        #
        # [X/Z, Y/Z, 1]
        #
        # Do NOT unit-normalize here.
        # ============================================================

        target_ray = torch.cat(
            [
                query_x_cam,
                query_y_cam,
                torch.ones_like(
                    query_x_cam
                ),
            ],
            dim=-1,
        )
        # [M,Q,3]


        # ============================================================
        # 12. Source camera XYZ
        #
        # Assuming z_used is optical-axis camera depth:
        #
        # X = (u-cx)/fx * Z
        # Y = (v-cy)/fy * Z
        # Z = Z
        # ============================================================

        source_xyz = torch.cat(
            [
                corr_x_cam * z_used,
                corr_y_cam * z_used,
                z_used,
            ],
            dim=-1,
        )
        # [M,Q,3]


        # Metric stabilization
        source_xyz_norm = (
            source_xyz
            / 80.0
        ).clamp(
            min=-2.0,
            max=2.0,
        )


        # ============================================================
        # 13. Intrinsic-normalized ray displacement
        # ============================================================

        ray_delta = torch.cat(
            [
                query_x_cam - corr_x_cam,
                query_y_cam - corr_y_cam,
            ],
            dim=-1,
        )
        # [M,Q,2]


        ray_delta = (
            ray_delta
            .clamp(
                min=-2.0,
                max=2.0,
            )
            / 2.0
        )


        # ============================================================
        # 14. CorrNet local feature map
        # ============================================================

        local_map = self.local_adaptor(
            enc_out
        )


        # ============================================================
        # 15. Query local feature
        # ============================================================

        query_local = self._sample_local_feature(
            local_map,
            query_uv_sbs,
        )
        # [M,Q,local_dim]


        # ============================================================
        # 16. Correspondence local feature
        # ============================================================

        corr_local = self._sample_local_feature(
            local_map,
            corr_uv_sbs,
        )
        # [M,Q,local_dim]


        # ============================================================
        # 17. Local matching feature
        #
        # query        = 32
        # corr         = 32
        # abs diff     = 32
        #
        # total        = 96
        # ============================================================

        local_match_feature = torch.cat(
            [
                query_local,
                corr_local,
                torch.abs(
                    query_local
                    - corr_local
                ),
            ],
            dim=-1,
        )


        # ============================================================
        # 18. Per-correspondence descriptor
        #
        # Legacy geometry                  7
        # Target ray                       3
        # Source XYZ                       3
        # Ray displacement                 2
        # Z reliability                    4
        # Adaptive Z gate                  1
        # Corr local matching             96
        #
        # Total                           116
        # ============================================================

        point_input = torch.cat(
            [
                geom_feature,            # 7

                target_ray,              # 3
                source_xyz_norm,         # 3
                ray_delta,               # 2

                z_reliability,           # 4
                z_gate,                  # 1

                local_match_feature,     # 96
            ],
            dim=-1,
        )
        # [M,Q,116]


        # Hard guard while V2 is being validated.
        if point_input.shape[-1] != 116:
            raise RuntimeError(
                '[CalibHead V2] '
                'K-aware point descriptor must be 116D, '
                f'got {point_input.shape}'
            )


        # ============================================================
        # 19. Shared point encoder
        # ============================================================

        point_feature = self.point_encoder(
            point_input
        )
        # [M,Q,64]


        # ============================================================
        # 20. Reliability scoring descriptor
        #
        # point feature       64
        # Z reliability        4
        # adaptive Z gate      1
        #
        # total               69
        # ============================================================

        score_input = torch.cat(
            [
                point_feature,
                z_reliability,
                z_gate,
            ],
            dim=-1,
        )
        # [M,Q,69]


        # ============================================================
        # 21. Rotation reliability
        # ============================================================

        rot_logits = self.rot_score_head(
            score_input
        )
        # [M,Q,1]


        rot_weight = self._masked_softmax(
            rot_logits,
            point_valid_mask,
        )
        # [M,Q]


        # ============================================================
        # 22. Rotation SET aggregation
        # ============================================================

        rot_set_feature = (
            rot_weight.unsqueeze(-1)
            * point_feature
        ).sum(
            dim=1
        )
        # [M,64]


        # ============================================================
        # 23. Global Corr context
        # ============================================================

        global_feature = (
            self.global_pool(
                enc_out
            )
            .flatten(1)
        )


        global_feature = self.global_encoder(
            global_feature
        )
        # [M,64]


        # ============================================================
        # 24. Rotation prediction FIRST
        # ============================================================

        rot_feature = torch.cat(
            [
                rot_set_feature,
                global_feature,
            ],
            dim=-1,
        )
        # [M,128]


        pred_rot = self.rot_head(
            rot_feature
        )
        # [M,3]


        # ============================================================
        # 25. Explicit Rotation Condition
        #
        # IMPORTANT:
        #
        # detach() intentionally prevents translation loss from
        # modifying the Rotation Head in this first experiment.
        #
        # We want to test:
        #
        #   "Does a known/predicted rotation estimate help translation?"
        #
        # without destabilizing the rotation branch.
        # ============================================================

        rot_condition_input = (
            pred_rot.detach()
            / self.rot_condition_scale_rad
        ).clamp(
            min=-2.0,
            max=2.0,
        )
        # [M,3]


        rot_condition = self.rot_condition_encoder(
            rot_condition_input
        )
        # [M,16]


        # ============================================================
        # 26. Rotation-conditioned Translation reliability
        #
        # Each correspondence sees the SAME camera-level rotation
        # estimate in addition to its own local geometry.
        # ============================================================

        rot_condition_per_point = (
            rot_condition
            .unsqueeze(1)
            .expand(
                -1,
                Q,
                -1,
            )
        )
        # [M,Q,16]


        trans_score_input = torch.cat(
            [
                score_input,                  # 69
                rot_condition_per_point,      # 16
            ],
            dim=-1,
        )
        # [M,Q,85]


        trans_logits = self.trans_score_head(
            trans_score_input
        )
        # [M,Q,1]


        trans_weight = self._masked_softmax(
            trans_logits,
            point_valid_mask,
        )
        # [M,Q]


        # ============================================================
        # 27. Translation SET aggregation
        # ============================================================

        trans_set_feature = (
            trans_weight.unsqueeze(-1)
            * point_feature
        ).sum(
            dim=1
        )
        # [M,64]


        # ============================================================
        # 28. Rotation-conditioned Translation feature
        #
        # Translation input:
        #
        #   trans set feature   64
        #   global Corr feature 64
        #   rotation condition  16
        #
        # total = 144
        # ============================================================

        trans_feature = torch.cat(
            [
                trans_set_feature,
                global_feature,
                rot_condition,
            ],
            dim=-1,
        )
        # [M,144]


        # ============================================================
        # 29. Translation prediction
        # ============================================================

        pred_trans = self.trans_head(
            trans_feature
        )
        # [M,3]


        # ============================================================
        # 30. Final 6-DoF
        # ============================================================

        pred_delta_6dof = torch.cat(
            [
                pred_rot,
                pred_trans,
            ],
            dim=-1,
        )
        # [M,6]


        # ============================================================
        # 27. No-valid-correspondence fallback
        #
        # If there is no real usable correspondence:
        #
        #     correction = identity / zero delta
        #
        # Do not let global Corr feature alone hallucinate a pose.
        # ============================================================

        has_valid_point = (
            point_valid_mask
            .any(
                dim=1,
                keepdim=True,
            )
        )


        pred_delta_6dof = torch.where(
            has_valid_point,
            pred_delta_6dof,
            torch.zeros_like(
                pred_delta_6dof
            ),
        )


        # ============================================================
        # 28. Diagnostics
        # ============================================================

        with torch.no_grad():

            valid_float = point_valid_mask.to(
                dtype=rot_weight.dtype
            )


            valid_count_per_sample = (
                valid_float
                .sum(
                    dim=1
                )
            )


            valid_count = (
                valid_count_per_sample
                .mean()
            )


            valid_ratio = (
                valid_float
                .mean()
            )


            has_valid = (
                valid_count_per_sample
                > 0
            )

            rot_condition_input_norm = (
                torch.linalg.vector_norm(
                    rot_condition_input,
                    dim=-1,
                )
                .mean()
            )

            pred_rot_norm_deg = (
                torch.linalg.vector_norm(
                    pred_rot.detach(),
                    dim=-1,
                )
                .mean()
                * (
                    180.0
                    / math.pi
                )
            )


            # --------------------------------------------------------
            # Reliability weight max
            # --------------------------------------------------------

            rot_weight_max = (
                rot_weight
                .max(
                    dim=1
                )
                .values
                .mean()
            )


            trans_weight_max = (
                trans_weight
                .max(
                    dim=1
                )
                .values
                .mean()
            )


            # --------------------------------------------------------
            # Effective correspondence count
            #
            # K_eff = 1 / sum(w_i^2)
            # --------------------------------------------------------

            rot_effective_k = (
                1.0
                /
                rot_weight
                .pow(2)
                .sum(
                    dim=1
                )
                .clamp_min(
                    1e-8
                )
            )


            trans_effective_k = (
                1.0
                /
                trans_weight
                .pow(2)
                .sum(
                    dim=1
                )
                .clamp_min(
                    1e-8
                )
            )


            if has_valid.any():

                rot_effective_k_mean = (
                    rot_effective_k[
                        has_valid
                    ]
                    .mean()
                )

                trans_effective_k_mean = (
                    trans_effective_k[
                        has_valid
                    ]
                    .mean()
                )

            else:

                rot_effective_k_mean = (
                    rot_weight
                    .new_tensor(0.0)
                )

                trans_effective_k_mean = (
                    trans_weight
                    .new_tensor(0.0)
                )


            # --------------------------------------------------------
            # Final Z gate actually used
            # --------------------------------------------------------

            z_gate_scalar = (
                z_gate
                .squeeze(-1)
            )


            z_gate_sum = (
                z_gate_scalar
                * valid_float
            ).sum(
                dim=1
            )


            z_gate_mean_per_sample = (
                z_gate_sum
                /
                valid_count_per_sample
                .clamp_min(
                    1.0
                )
            )


            if has_valid.any():

                z_gate_mean = (
                    z_gate_mean_per_sample[
                        has_valid
                    ]
                    .mean()
                )

            else:

                z_gate_mean = (
                    z_gate
                    .new_tensor(0.0)
                )


            # --------------------------------------------------------
            # Pure learned Z gate
            #
            # Excludes forced gate=0 / gate=1 cases.
            # --------------------------------------------------------

            learnable_gate_float = (
                learnable_gate_mask
                .to(
                    dtype=z_gate_learned.dtype
                )
            )


            learnable_gate_count_per_sample = (
                learnable_gate_float
                .sum(
                    dim=1
                )
            )


            learnable_gate_count = (
                learnable_gate_count_per_sample
                .mean()
            )


            learned_gate_sum = (
                z_gate_learned
                .squeeze(-1)
                * learnable_gate_float
            ).sum(
                dim=1
            )


            learned_gate_mean_per_sample = (
                learned_gate_sum
                /
                learnable_gate_count_per_sample
                .clamp_min(
                    1.0
                )
            )


            has_learnable_gate = (
                learnable_gate_count_per_sample
                > 0
            )


            if has_learnable_gate.any():

                z_gate_learned_mean = (
                    learned_gate_mean_per_sample[
                        has_learnable_gate
                    ]
                    .mean()
                )

            else:

                z_gate_learned_mean = (
                    z_gate_learned
                    .new_tensor(0.0)
                )
            
            # ============================================================
            # Learned gate distribution diagnostics
            #
            # Mean alone is insufficient:
            #
            #   [0.2, 0.8] -> mean = 0.5
            #
            # Therefore measure:
            #
            #   std
            #   mean absolute deviation from 0.5
            # ============================================================

            learned_gate_values = (
                z_gate_learned
                .squeeze(-1)[
                    learnable_gate_mask
                ]
            )


            if learned_gate_values.numel() > 0:

                z_gate_learned_std = (
                    learned_gate_values
                    .std(
                        unbiased=False
                    )
                )


                z_gate_learned_absdev = (
                    torch.abs(
                        learned_gate_values
                        - 0.5
                    )
                    .mean()
                )


                z_gate_low_frac = (
                    (
                        learned_gate_values
                        < 0.4
                    )
                    .float()
                    .mean()
                )


                z_gate_high_frac = (
                    (
                        learned_gate_values
                        > 0.6
                    )
                    .float()
                    .mean()
                )


            else:

                z_gate_learned_std = (
                    z_gate_learned
                    .new_tensor(0.0)
                )

                z_gate_learned_absdev = (
                    z_gate_learned
                    .new_tensor(0.0)
                )

                z_gate_low_frac = (
                    z_gate_learned
                    .new_tensor(0.0)
                )

                z_gate_high_frac = (
                    z_gate_learned
                    .new_tensor(0.0)
                )


            # --------------------------------------------------------
            # Raw Z vs Pred Z gap
            # --------------------------------------------------------

            z_gap_meter = (
                torch.abs(
                    z_pred - z_raw
                )
                .squeeze(-1)
            )


            z_gap_sum = (
                z_gap_meter
                * valid_float
            ).sum(
                dim=1
            )


            z_gap_mean_per_sample = (
                z_gap_sum
                /
                valid_count_per_sample
                .clamp_min(
                    1.0
                )
            )


            if has_valid.any():

                z_gap_mean = (
                    z_gap_mean_per_sample[
                        has_valid
                    ]
                    .mean()
                )

            else:

                z_gap_mean = (
                    z_gap_meter
                    .new_tensor(0.0)
                )


            # --------------------------------------------------------
            # K-aware ray displacement diagnostic
            # --------------------------------------------------------

            ray_delta_norm = (
                torch.linalg.vector_norm(
                    ray_delta,
                    dim=-1,
                )
            )


            ray_delta_sum = (
                ray_delta_norm
                * valid_float
            ).sum(
                dim=1
            )


            ray_delta_mean_per_sample = (
                ray_delta_sum
                /
                valid_count_per_sample
                .clamp_min(
                    1.0
                )
            )


            if has_valid.any():

                ray_delta_mean = (
                    ray_delta_mean_per_sample[
                        has_valid
                    ]
                    .mean()
                )

            else:

                ray_delta_mean = (
                    ray_delta_norm
                    .new_tensor(0.0)
                )


        # ============================================================
        # 29. Diagnostic dictionary
        # ============================================================

        diagnostics = {

            # --------------------------------------------------------
            # Duplicate-mask diagnostics
            # --------------------------------------------------------

            'calib_valid_count':
                valid_count,

            'calib_valid_ratio':
                valid_ratio,


            # --------------------------------------------------------
            # Adaptive Z diagnostics
            # --------------------------------------------------------

            'calib_z_gate_mean':
                z_gate_mean,

            'calib_z_gate_learned_mean':
                z_gate_learned_mean,

            'calib_z_gate_learnable_count':
                learnable_gate_count,

            'calib_z_pred_raw_gap_m':
                z_gap_mean,
            
            'calib_z_gate_learned_std':
                z_gate_learned_std,

            'calib_z_gate_learned_absdev':
                z_gate_learned_absdev,

            'calib_z_gate_low_frac':
                z_gate_low_frac,

            'calib_z_gate_high_frac':
                z_gate_high_frac,

            # --------------------------------------------------------
            # K-aware geometry diagnostic
            # --------------------------------------------------------

            'calib_ray_delta_mean':
                ray_delta_mean,


            # --------------------------------------------------------
            # Rotation reliability
            # --------------------------------------------------------

            'calib_rot_weight_max':
                rot_weight_max,

            'calib_rot_effective_k':
                rot_effective_k_mean,
            
            'calib_rot_condition_norm':
                rot_condition_input_norm,

            'calib_pred_rot_norm_deg':
                pred_rot_norm_deg,


            # --------------------------------------------------------
            # Translation reliability
            # --------------------------------------------------------

            'calib_trans_weight_max':
                trans_weight_max,

            'calib_trans_effective_k':
                trans_effective_k_mean,
        }


        return (
            pred_delta_6dof,
            diagnostics,
        )