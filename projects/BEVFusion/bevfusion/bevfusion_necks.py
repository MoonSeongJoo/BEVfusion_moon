# modify from https://github.com/mit-han-lab/bevfusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint 
from mmengine import print_log   


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
            init_cfg=None,
    ) -> None:
        # <-- ✨ 2. super()에 init_cfg 전달
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # --- ✨ 3. 가중치 수동 로드 로직 (모든 레이어 생성 후) ---
        if self.init_cfg and self.init_cfg.get('type') == 'Pretrained':
            checkpoint_path = self.init_cfg.get('checkpoint')
            if checkpoint_path:
                print_log(f'Manually loading checkpoint for GeneralizedLSSFPN from: {checkpoint_path}', logger='current')
                
                # ⭐️ (중요) 'self' (GeneralizedLSSFPN 인스턴스)에 로드합니다.
                load_checkpoint(
                    self, 
                    checkpoint_path, 
                    map_location='cpu', 
                    strict=False, # True로 하면 키가 정확히 일치해야 함
                    
                    # ⭐️ (중요) 체크포인트의 접두사에 맞게 수정하세요.
                    # 예: 체크포인트 키가 'img_neck.lateral_convs...' 라면
                    revise_keys=[('^img_neck\\.', '')]
                    # 예: 체크포인트 키가 'neck.lateral_convs...' 라면
                    # revise_keys=[('^neck\\.', '')]
                    # 예: 접두사가 없다면 이 'revise_keys' 라인을 삭제하거나 주석 처리
                )
            else:
                print_log('No checkpoint path in init_cfg for GeneralizedLSSFPN.', logger='current', level='WARNING')

    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)
