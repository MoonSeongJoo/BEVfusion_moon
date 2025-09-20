import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ..utils import debug_utils, constants, utils
from .misc_moon import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone_moon import build_backbone
from .position_encoding_moon import build_position_encoding
from .transformer_moon import build_transformer
from .position_encoding_moon import NerfPositionalEncoding, MLP


class COTR(nn.Module):

    def __init__(self, backbone, transformer, sine_type='lin_sine'):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_proj = NerfPositionalEncoding(hidden_dim // 4, sine_type)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, queries):
#         print ("sampels_shape" , samples.shape)
#         print ("queries_shape" , queries.shape)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
#         print ("src_shape" , src.shape)
#         print ("mask_shape" , mask.shape)
#         print ("queries_shape" , queries.shape)
#         print("pos_shape" , pos.shape)
        
        _b, _q, _ = queries.shape
        queries = queries.reshape(-1, 2)
        queries = self.query_proj(queries).reshape(_b, _q, -1)
        queries = queries.permute(1, 0, 2)
        tr_input= self.input_proj(src)
#         print ("tr_input", tr_input.shape)
#         print ("queries_after_shape" , queries.shape)
        hs = self.transformer(tr_input, mask, queries, pos[-1])[0]
        outputs_corr = self.corr_embed(hs)
        out = {'pred_corrs': outputs_corr[-1]}
        return out


def build(args):
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = COTR(
        backbone,
        transformer,
        sine_type=args.position_embedding,
    )
    return model
