import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from ..utils import debug_utils, constants, utils
from .misc_moon import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone_moon_Ver5 import build_backbone
from .position_encoding_moon import build_position_encoding
from .transformer_moon import build_transformer
from .position_encoding_moon import NerfPositionalEncoding, MLP

class COTR(nn.Module):

    def __init__(self, backbone, transformer, sine_type='lin_sine',return_local_layer2=False,):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.corr_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        # self.query_proj = NerfPositionalEncoding(hidden_dim // 6, sine_type) # uvz points 대응할때
        self.query_proj = NerfPositionalEncoding(hidden_dim // 4, sine_type) # uv points 대응할때 
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.return_local_layer2 = (
            return_local_layer2
        )

    def forward(self, samples: NestedTensor, queries):
#         print ("sampels_shape" , samples.shape)
        # print ("queries_shape1" , queries.shape)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            features, pos = (
                self.backbone(
                    samples
                )
            )

            if self.return_local_layer2:

                if len(features) != 2:

                    raise RuntimeError(
                        '[COTR] Expected layer2 + layer3 '
                        f'but got {len(features)} features.'
                    )


                # ========================================================
                # High-resolution local feature
                #
                # Expected:
                # [B,512,24,160]
                # ========================================================

                local_feature = (
                    features[0].tensors
                )


                # ========================================================
                # ORIGINAL coarse COTR feature
                #
                # Expected:
                # layer3 [B,1024,12,80]
                # ========================================================

                coarse_feature = (
                    features[1]
                )


                coarse_pos = (
                    pos[1]
                )


            else:

                local_feature = None

                coarse_feature = (
                    features[-1]
                )

                coarse_pos = (
                    pos[-1]
                )


            src, mask = (
                coarse_feature.decompose()
            )

            assert mask is not None
        assert mask is not None

        _b, _q, _ = queries.shape
        queries = queries.reshape(-1, 2)
        # queries = queries.reshape(-1, 3)
        queries = self.query_proj(queries)
        queries = queries.reshape(_b, _q, -1)
        queries = queries.permute(1, 0, 2)
        # queries = torch.tensor(queries ,dtype=torch.float32)
        # queries_clone = queries.clone().detach()
        queries_clone = queries.clone()
        tr_input= self.input_proj(src)
        hs, enc_out = self.transformer(
            tr_input,
            mask,
            queries_clone,
            coarse_pos,
        )
        outputs_corr = self.corr_embed(hs)[-1]
        corr_out = outputs_corr
        return (
            corr_out,
            enc_out,
            local_feature,
        )

def build(args):
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = COTR(

        backbone,

        transformer,

        sine_type=
            args.position_embedding,

        return_local_layer2=
            getattr(
                args,
                'return_local_layer2',
                False,
            ),
    )
    return model
