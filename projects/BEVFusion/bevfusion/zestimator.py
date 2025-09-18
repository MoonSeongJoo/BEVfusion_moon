import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule 

@MODELS.register_module()
class ZEstimator(BaseModule):
    def __init__(self, enc_channels=312, uv_dim=2, hidden_dim=512, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # enc_out 특징 압축
        self.enc_adaptor = nn.Sequential(
            nn.Conv2d(enc_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # UV 좌표 임베딩
        self.uv_embed = nn.Linear(uv_dim, 64)
        
        # ✨ 1. bbox_feats가 없어졌으므로 융합 레이어의 입력 차원 수정 (128 + 128 + 64 -> 128 + 64)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # 신뢰도 예측을 위한 작은 MLP (옵션)
        self.confidence_predictor = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def estimate_lidar_confidence(self, z_depth_real):
        # z_depth_real은 [Total_Queries] 형태의 1D 텐서
        # 신뢰도를 예측하기 위해 unsqueeze로 채널 차원 추가
        return self.confidence_predictor(z_depth_real.unsqueeze(-1)).squeeze(-1)

    def forward(self, uv, depth_map, enc_out):
        """
        Args:
            uv (torch.Tensor): [N*V, Q, 2] 형태 (u, v 좌표)
            depth_map (torch.Tensor): [N, V, H, W] 형태
            enc_out (torch.Tensor): [N*V, C, H_feat, W_feat] 형태
        """
        # ✨ 2. 새로운 입력 형태에 맞춰 변수 준비
        NV, Q, _ = uv.shape
        N, V, H, W = depth_map.shape
        device = uv.device

        # (N*V, Q, 2) -> (N*V*Q, 2) 형태로 평탄화
        uv_flat = uv.view(NV * Q, 2)

        # depth_map도 (N*V, H, W) 형태로 변경
        depth_map_reshaped = depth_map.view(NV, H, W)
        
        # ✨ 3. cam_ids를 동적으로 생성
        # [0, 0, ..., 1, 1, ..., 11, 11, ...] 형태의 텐서 생성 (길이: N*V*Q)
        cam_ids = torch.arange(NV, device=device).unsqueeze(1).expand(NV, Q).reshape(-1)

        # 4. enc_out 특징 처리 (기존과 유사)
        enc_reduced = self.enc_adaptor(enc_out).squeeze(-1).squeeze(-1)  # [N*V, 128]
        
        # 객체별 enc 특징 선택
        object_enc = enc_reduced[cam_ids]  # [N*V*Q, 128]
        
        # ✨ 5. bbox_feats 처리 로직은 완전히 제거
        
        # 6. UV 좌표 처리 (uv_flat 사용)
        uv_embedded = self.uv_embed(uv_flat)  # [N*V*Q, 64]
        
        # 7. 특징 융합 (bbox_reduced 제외)
        combined = torch.cat([object_enc, uv_embedded], dim=1)
        fused = self.fusion(combined)
        
        # 8. 깊이 예측
        z_estimated_real = self.depth_predictor(fused).squeeze(-1) # [N*V*Q]
        
        # 9. 실제 LiDAR 깊이 조회 (평탄화된 좌표 사용)
        u_coords = uv_flat[:, 0].clamp(0, W - 1).long()
        v_coords = uv_flat[:, 1].clamp(0, H - 1).long()
        z_depth_real = depth_map_reshaped[cam_ids, v_coords, u_coords] # [N*V*Q]
        
        # 10. 신뢰도 기반 융합 (기존과 동일)
        lidar_confidence = self.estimate_lidar_confidence(z_depth_real)
        valid_lidar_mask = (z_depth_real > 0)
        lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()
        
        z_final_real = (
            lidar_confidence_adjusted * z_depth_real +
            (1 - lidar_confidence_adjusted) * z_estimated_real
        )
        
        # ✨ 11. 최종 출력을 입력 uv 형태와 유사하게 (N*V, Q, 1)로 복원
        return {
            'depth': z_final_real.view(NV, Q, 1),
            'z_lidar_real': z_depth_real.view(NV, Q, 1),
            'confidence': lidar_confidence.view(NV, Q, 1),
            'z_estimated_real': z_estimated_real.view(NV, Q, 1),
        }