# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet3d.registry import MODELS
# from mmengine.model import BaseModule 

# @MODELS.register_module()
# class ZEstimator(BaseModule):
#     def __init__(self, enc_channels=312, uv_dim=2, hidden_dim=512, init_cfg=None):
#         super().__init__(init_cfg=init_cfg)
#         # enc_out 특징 압축
#         self.enc_adaptor = nn.Sequential(
#             nn.Conv2d(enc_channels, 128, kernel_size=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
        
#         # UV 좌표 임베딩
#         self.uv_embed = nn.Linear(uv_dim, 64)
        
#         # ✨ 1. bbox_feats가 없어졌으므로 융합 레이어의 입력 차원 수정 (128 + 128 + 64 -> 128 + 64)
#         self.fusion = nn.Sequential(
#             nn.Linear(128 + 64, hidden_dim),
#             nn.LeakyReLU(negative_slope=0.01),
#             # nn.ReLU(),
#             nn.LayerNorm(hidden_dim)
#         )
#         self.depth_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.LeakyReLU(negative_slope=0.01),
#             # nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.ReLU()  # ✨ FIX: 음수 값을 0으로 만들기 위해 ReLU 추가
#         )
#         # 신뢰도 예측을 위한 작은 MLP (옵션)
#         self.confidence_predictor = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def estimate_lidar_confidence(self, z_depth_real):
#         # z_depth_real은 [Total_Queries] 형태의 1D 텐서
#         # 신뢰도를 예측하기 위해 unsqueeze로 채널 차원 추가
#         return self.confidence_predictor(z_depth_real.unsqueeze(-1)).squeeze(-1)

#     def forward(self, uv, depth_map, enc_out):
#         """
#         Args:
#             uv (torch.Tensor): [N*V, Q, 2] 형태 (u, v 좌표)
#             depth_map (torch.Tensor): [N, V, H, W] 형태
#             enc_out (torch.Tensor): [N*V, C, H_feat, W_feat] 형태
#         """
#         # ✨ 2. 새로운 입력 형태에 맞춰 변수 준비
#         NV, Q, _ = uv.shape
#         N, V, H, W = depth_map.shape
#         device = uv.device

#         # (N*V, Q, 2) -> (N*V*Q, 2) 형태로 평탄화
#         uv_flat = uv.view(NV * Q, 2)

#         # depth_map도 (N*V, H, W) 형태로 변경
#         depth_map_reshaped = depth_map.view(NV, H, W)
        
#         # ✨ 3. cam_ids를 동적으로 생성
#         # [0, 0, ..., 1, 1, ..., 11, 11, ...] 형태의 텐서 생성 (길이: N*V*Q)
#         cam_ids = torch.arange(NV, device=device).unsqueeze(1).expand(NV, Q).reshape(-1)

#         # 4. enc_out 특징 처리 (기존과 유사)
#         enc_reduced = self.enc_adaptor(enc_out).squeeze(-1).squeeze(-1)  # [N*V, 128]
        
#         # 객체별 enc 특징 선택
#         object_enc = enc_reduced[cam_ids]  # [N*V*Q, 128]
        
#         # ✨ 5. bbox_feats 처리 로직은 완전히 제거
        
#         # 6. UV 좌표 처리 (uv_flat 사용)
#         uv_embedded = self.uv_embed(uv_flat)  # [N*V*Q, 64]
        
#         # 7. 특징 융합 (bbox_reduced 제외)
#         combined = torch.cat([object_enc, uv_embedded], dim=1)
#         fused = self.fusion(combined)
        
#         # 8. 깊이 예측
#         z_estimated_real = self.depth_predictor(fused).squeeze(-1) # [N*V*Q]
        
#         # 9. 실제 LiDAR 깊이 조회 (평탄화된 좌표 사용)
#         u_coords = uv_flat[:, 0].clamp(0, W - 1).long()
#         v_coords = uv_flat[:, 1].clamp(0, H - 1).long()
#         z_depth_real = depth_map_reshaped[cam_ids, v_coords, u_coords] # [N*V*Q]
        
#         # 10. 신뢰도 기반 융합 (기존과 동일)
#         lidar_confidence = self.estimate_lidar_confidence(z_depth_real)
#         valid_lidar_mask = (z_depth_real > 0)
#         lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()
        
#         z_final_real = (
#             lidar_confidence_adjusted * z_depth_real +
#             (1 - lidar_confidence_adjusted) * z_estimated_real
#         )
        
#         # ✨ 11. 최종 출력을 입력 uv 형태와 유사하게 (N*V, Q, 1)로 복원
#         return {
#             'depth': z_final_real.view(NV, Q, 1),
#             'z_lidar_real': z_depth_real.view(NV, Q, 1),
#             'confidence': lidar_confidence.view(NV, Q, 1),
#             'z_estimated_real': z_estimated_real.view(NV, Q, 1),
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class ZEstimator(BaseModule):
    """
    글로벌 컨텍스트, 로컬 특징, 위치 정보를 모두 융합하여 깊이를 추정하는 모듈.
    """
    def __init__(self, enc_channels=312, local_channels=312, uv_dim=2, hidden_dim=512, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 1. 글로벌 컨텍스트 추출기
        self.enc_adaptor = nn.Sequential(
            nn.Conv2d(enc_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # # 2. 위치 정보 임베딩
        self.uv_embed = nn.Linear(uv_dim, 64)
        
        # 3. 모든 특징을 융합하는 레이어
        # ✨ FIX: 로컬 특징(local_channels)이 추가되었으므로 입력 차원 확장
        fusion_in_channels = 128 + local_channels + 64  # Global + Local + Positional
        # fusion_in_channels = 128 + local_channels  # Global + Local : positional 빼고 해보기
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_channels, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 깊이 예측기
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 1),
            # nn.ReLU()  # 깊이는 항상 0 이상이어야 하므로 최종 출력은 ReLU
            nn.Sigmoid(),
        )
        
        # # 5. LiDAR 포인트 신뢰도 예측기 (기존과 동일)
        # self.confidence_predictor = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
        # ✨✨✨ START: 최종 레이어 편향(Bias) 초기화 ✨✨✨
        # self.depth_predictor의 마지막 Linear 레이어(-2 인덱스)를 타겟으로 합니다.
        # 편향(bias)을 작은 양수(예: 0.1)로 초기화합니다.
        # 이렇게 하면 초기 출력이 음수가 될 확률이 크게 줄어듭니다.
        torch.nn.init.constant_(self.depth_predictor[-2].bias.data, 0.1)
        # ✨✨✨ END: 편향 초기화 ✨✨✨

    def estimate_lidar_confidence(self, z_depth_real):
        return self.confidence_predictor(z_depth_real.unsqueeze(-1)).squeeze(-1)

    # def forward(self, uv, depth_map, enc_out):
    #     """
    #     Args:
    #         uv (torch.Tensor): [N*V, Q, 2] 형태. 각 좌표는 [0, 1] 범위로 정규화되어야 함.
    #         depth_map (torch.Tensor): [N, V, H, W] 형태
    #         enc_out (torch.Tensor): [N*V, C, H_feat, W_feat] 형태
    #     """
    #     # --- 입력 변수 준비 ---
    #     NV, Q, _ = uv.shape
    #     N, V, H, W = depth_map.shape
    #     C_local = enc_out.shape[1] # 로컬 특징의 채널 수
    #     device = uv.device

    #     uv_flat = uv.view(NV * Q, 2)
    #     depth_map_reshaped = depth_map.view(NV, H, W)
    #     cam_ids = torch.arange(NV, device=device).unsqueeze(1).expand(NV, Q).reshape(-1)

    #     # --- 1. 글로벌 특징(Global Feature) 추출 ---
    #     enc_reduced = self.enc_adaptor(enc_out).squeeze(-1).squeeze(-1)  # [N*V, 128]
    #     global_features = enc_reduced[cam_ids]  # [N*V*Q, 128]
        
    #     # --- 2. 로컬 특징(Local Feature) 샘플링 ---
    #     # grid_sample을 위해 uv 좌표를 [0, 1] -> [-1, 1] 범위로 변환
    #     uv_clone =uv.clone()
    #     uv_clone[...,0] =uv_clone[...,0]/1600
    #     uv_clone[...,1] =uv_clone[...,1]/900
    #     grid = uv_clone * 2 - 1
        
    #     # grid_sample의 입력 형태에 맞게 grid 차원 변경: [N*V, Q, 2] -> [N*V, 1, Q, 2]
    #     grid = grid.unsqueeze(1)
        
    #     # enc_out 피처맵에서 각 uv 좌표에 해당하는 로컬 특징을 직접 샘플링
    #     local_features_sampled = F.grid_sample(enc_out, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    #     # 결과 shape: [N*V, C_local, 1, Q]
        
    #     # 융합을 위해 모양 변경: [N*V, C_local, 1, Q] -> [N*V*Q, C_local]
    #     local_features_flat = local_features_sampled.squeeze(2).permute(0, 2, 1).reshape(NV * Q, C_local)
        
    #     # # --- 3. 위치 정보(Positional Feature) 임베딩 ---
    #     uv_embedded = self.uv_embed(uv_flat)  # [N*V*Q, 64]
        
    #     # --- 4. 모든 특징 융합 ---
    #     combined = torch.cat([global_features, local_features_flat, uv_embedded], dim=1)
    #     # combined = torch.cat([global_features, local_features_flat], dim=1)
    #     fused = self.fusion(combined)
        
    #     # --- 5. 깊이 예측 및 후처리 (기존과 동일) ---
    #     raw_prediction = self.depth_predictor(fused).squeeze(-1) # [N*V*Q]
    #     z_estimated_real = raw_prediction * 80.0
        
    #     # u_coords = (uv_flat[:, 0] * (W - 1)).round().long().clamp(0, W - 1)
    #     # v_coords = (uv_flat[:, 1] * (H - 1)).round().long().clamp(0, H - 1)
    #     u_coords = uv_flat[:, 0].round().long().clamp(0, W - 1)
    #     v_coords = uv_flat[:, 1].round().long().clamp(0, H - 1)
    #     z_depth_real = depth_map_reshaped[cam_ids, v_coords, u_coords] # [N*V*Q]
        
    #     # lidar_confidence = self.estimate_lidar_confidence(z_depth_real)
    #     # valid_lidar_mask = (z_depth_real > 0)
    #     # lidar_confidence_adjusted = lidar_confidence * valid_lidar_mask.float()

    #     lidar_confidence_adjusted = (z_depth_real > 0).float()
    #     z_final_real = (
    #         lidar_confidence_adjusted * z_depth_real +
    #         (1 - lidar_confidence_adjusted) * z_estimated_real
    #     )
        
    #     # --- 최종 출력 ---
    #     return {
    #         'depth': z_final_real.view(NV, Q, 1),
    #         'z_lidar_real': z_depth_real.view(NV, Q, 1),
    #         'confidence': lidar_confidence_adjusted.view(NV, Q, 1),
    #         'z_estimated_real': z_estimated_real.view(NV, Q, 1),
    #     }


    # forward 시그니처를 두 개의 좌표로 변경
    def forward(self, uv_sbs_normalized, uv_orig_pixels, depth_map, enc_out):
        """
        Args:
            uv_sbs_normalized (torch.Tensor): [N*V, Q, 2] (Normalized [0.5, 1] for 192x1280 SBS map)
            uv_orig_pixels (torch.Tensor):    [N*V, Q, 2] (Pixel coordinates for 1600x900 original map)
            depth_map (torch.Tensor):         [N, V, H, W] (H=900, W=1600)
            enc_out (torch.Tensor):           [N*V, C, H_feat, W_feat] (from 192x1280 map)
        """
        # --- 입력 변수 준비 ---
        NV, Q, _ = uv_sbs_normalized.shape
        N, H, W = depth_map.shape # H=900, W=1600
        C_local = enc_out.shape[1]
        device = uv_sbs_normalized.device
        cam_ids = torch.arange(NV, device=device).unsqueeze(1).expand(NV, Q).reshape(-1)

        # --- 1. 글로벌 특징 (Global Feature) 추출 (동일) ---
        enc_reduced = self.enc_adaptor(enc_out).squeeze(-1).squeeze(-1)
        global_features = enc_reduced[cam_ids]
        
        # --- 2. 로컬 특징(Local Feature) 샘플링 (✨ 수정됨 ✨) ---
        # 'enc_out'을 샘플링하기 위해 'uv_sbs_normalized' 사용
        grid_sbs = uv_sbs_normalized * 2.0 - 1.0  # [0.5, 1] -> [0, 1] (x), [0, 1] -> [-1, 1] (y)
        grid_sbs = grid_sbs.unsqueeze(1) # [N*V, 1, Q, 2]
        
        local_features_sampled = F.grid_sample(
            enc_out, grid_sbs, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        local_features_flat = local_features_sampled.squeeze(2).permute(0, 2, 1).reshape(NV * Q, C_local)
        
        # --- 3. 위치 정보(Positional Feature) 임베딩 (✨ 수정됨 ✨) ---
        # 특징 맵 기준의 'uv_sbs_normalized'를 임베딩
        uv_sbs_flat = uv_sbs_normalized.view(NV * Q, 2)
        uv_embedded = self.uv_embed(uv_sbs_flat)
        
        # --- 4. 모든 특징 융합 (동일) ---
        combined = torch.cat([global_features, local_features_flat, uv_embedded], dim=1)
        fused = self.fusion(combined)
        
        # --- 5. 깊이 예측 및 GT 샘플링 (✨ 수정됨 ✨) ---
        raw_prediction = self.depth_predictor(fused).squeeze(-1)
        z_estimated_real = raw_prediction * 80.0
        
        # 'depth_map'을 샘플링하기 위해 'uv_orig_pixels' 사용
        depth_map_reshaped = depth_map.view(NV, H, W)
        uv_orig_flat = uv_orig_pixels.view(NV * Q, 2)
        
        # 픽셀 좌표를 정수형으로 변환
        u_coords = uv_orig_flat[:, 0].round().long().clamp(0, W - 1) # W = 1600
        v_coords = uv_orig_flat[:, 1].round().long().clamp(0, H - 1) # H = 900
        
        z_depth_real = depth_map_reshaped[cam_ids, v_coords, u_coords]
        
        # ... (이후 로직은 동일) ...
        lidar_confidence_adjusted = (z_depth_real > 0).float()
        z_final_real = (
            lidar_confidence_adjusted * z_depth_real +
            (1 - lidar_confidence_adjusted) * z_estimated_real
        )
        
        return {
            'depth': z_final_real.view(NV, Q, 1),
            'z_lidar_real': z_depth_real.view(NV, Q, 1),
            'confidence': lidar_confidence_adjusted.view(NV, Q, 1),
            'z_estimated_real': z_estimated_real.view(NV, Q, 1),
        }