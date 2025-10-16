import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

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