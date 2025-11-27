import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log

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

        # --- ✨ 2. 가중치 수동 로드 로직 (모든 레이어 생성 후) ---
        if self.init_cfg and self.init_cfg.get('type') == 'Pretrained':
            checkpoint_path = self.init_cfg.get('checkpoint')
            if checkpoint_path:
                print_log(f'Manually loading checkpoint for ZEstimator from: {checkpoint_path}', logger='current')
                
                # ⭐️ (중요) 'self' (ZEstimator 인스턴스)에 로드합니다.
                load_checkpoint(
                    self, 
                    checkpoint_path, 
                    map_location='cpu', 
                    strict=False,
                    
                    # ⭐️ (중요) 체크포인트의 접두사에 맞게 수정하세요.
                    # 예: 체크포인트 키가 'z_estimator.enc_adaptor...' 라면
                    revise_keys=[('^z_estimator\\.', '')]
                    # 예: 접두사가 없다면 이 'revise_keys' 라인을 삭제
                )
            else:
                print_log('No checkpoint path in init_cfg for ZEstimator.', logger='current', level='WARNING')

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