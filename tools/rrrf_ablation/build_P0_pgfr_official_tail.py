#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one shared P0 checkpoint for independent PGFR-Det RRRF A-E ablations.

P0 tensor provenance:
  PGFR    : all matching weights/buffers except the groups below
  Official: bbox_head.decoder.* and bbox_head.prediction_heads.*
  Fresh   : feat_projector and Coarse/Refined/Local Proposal modules

Run in the user's mmdetection3d/MMEngine environment and repository root.
This script creates a checkpoint; it does NOT train a model or edit source code.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

OFFICIAL_PREFIXES = (
    'bbox_head.decoder.',
    'bbox_head.prediction_heads.',
)

# The prefix matches both parameters and buffers, if any.
FRESH_PREFIXES = (
    'feat_projector.',
    'bbox_head.fusion_cross_attention.',
    'bbox_head.fusion_ffn.',
    'bbox_head.fusion_norm1.',
    'bbox_head.fusion_norm2.',
    'bbox_head.camera_proposal_pos_embedding.',
    'bbox_head.bev_query_pos_embedding.',
    'bbox_head.refined_attention.',
    'bbox_head.refined_ffn.',
    'bbox_head.refined_norm1.',
    'bbox_head.refined_norm2.',
    'bbox_head.rrrf_query_selector.',
    # These legacy heads are unused in feature_refine but are reset to avoid
    # inheriting RRRF-related parameters from the PGFR best checkpoint.
    'bbox_head.calibration_predictor.',
    'bbox_head.residual_',
    'bbox_head.refined_fusion_aux_head.',
)

REQUIRED_PREFIXES = (
    'corr.', 'z_estimator.', 'calib_head.',
    'bbox_head.shared_conv.', 'bbox_head.heatmap_head.',
    'bbox_head.class_encoding.',
    'bbox_head.decoder.', 'bbox_head.prediction_heads.',
    'feat_projector.',
    'bbox_head.fusion_cross_attention.',
    'bbox_head.refined_attention.',
    'bbox_head.rrrf_query_selector.',
)


def load_state_dict(path):
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError('Checkpoint is not a dict: {}'.format(path))
    state = obj.get('state_dict', obj)
    if not isinstance(state, dict):
        raise TypeError('Invalid state_dict in {}'.format(path))
    normalized = {}
    for name, value in state.items():
        key = name[7:] if name.startswith('module.') else name
        if key in normalized:
            raise ValueError('Duplicate normalized state_dict key: {}'.format(key))
        if not isinstance(value, torch.Tensor):
            raise TypeError('Expected tensor for key: {}'.format(key))
        normalized[key] = value
    return normalized


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pgfr', required=True, help='PGFR_Det_final_model.pth')
    ap.add_argument('--official', required=True, help='Official BEVFusion checkpoint')
    ap.add_argument('--config', required=True,
                    help='PGFR feature_refine config with Local Proposal selector enabled')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20261106)
    args = ap.parse_args()

    # Imports here let `--help` work without importing the model stack.
    from mmengine.config import Config
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion  # noqa: F401, registers project components

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_all_modules(init_default_scope=True)

    config = Config.fromfile(args.config)
    head_config = config.model.get('bbox_head', {})
    if head_config.get('rrrf_mode') != 'feature_refine':
        raise ValueError('P0 build config must have rrrf_mode=feature_refine')
    if not head_config.get('rrrf_query_select_enabled', False):
        raise ValueError('P0 build config must enable the Local Proposal selector')
    if head_config.get('rrrf_adapter_enabled', False):
        raise ValueError('P0 build config must disable the residual feature adapter')

    print('[BUILD] Constructing fresh model on CPU for common RRRF initialization')
    model = MODELS.build(config.model)
    fresh_sd = model.state_dict()
    pgfr_sd = load_state_dict(args.pgfr)
    official_sd = load_state_dict(args.official)

    problems = []
    for prefix in REQUIRED_PREFIXES:
        count = sum(key.startswith(prefix) for key in fresh_sd)
        if count == 0:
            problems.append('MISSING MODEL PREFIX: ' + prefix)

    merged = {}
    owners = {}
    counts = Counter()
    for key, expected in fresh_sd.items():
        if key.startswith(OFFICIAL_PREFIXES):
            origin, donor = 'Official', official_sd
        elif key.startswith(FRESH_PREFIXES):
            origin, donor = 'Fresh', fresh_sd
        else:
            origin, donor = 'PGFR', pgfr_sd

        if key not in donor:
            problems.append('MISSING [{}] {}'.format(origin, key))
            continue
        value = donor[key]
        if expected.shape != value.shape:
            problems.append('SHAPE [{}] {} target={} source={}'.format(
                origin, key, tuple(expected.shape), tuple(value.shape)))
            continue
        merged[key] = value.detach().cpu().clone()
        owners[key] = origin
        counts[origin] += 1

    # Do not silently omit any official decoder/head tensors.
    for key in official_sd:
        if key.startswith(OFFICIAL_PREFIXES) and key not in fresh_sd:
            problems.append('EXTRA Official decoder/head key: ' + key)

    if counts['Official'] != 84:
        problems.append('Expected 84 Official decoder/head tensors based on the '
                        'user inspection, got {}'.format(counts['Official']))

    if problems:
        print('\n'.join('[ERROR] ' + p for p in problems[:80]))
        raise RuntimeError('P0 NOT saved: {} incompatible keys'.format(len(problems)))

    # Numerical provenance equality for every non-fresh key.
    for key, origin in owners.items():
        if origin == 'PGFR' and not torch.equal(merged[key], pgfr_sd[key]):
            raise AssertionError('PGFR provenance mismatch: ' + key)
        if origin == 'Official' and not torch.equal(merged[key], official_sd[key]):
            raise AssertionError('Official provenance mismatch: ' + key)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'meta': {
            'purpose': 'RRRF A-E common P0, not a trained ablation result',
            'pgfr_source': str(Path(args.pgfr).resolve()),
            'official_source': str(Path(args.official).resolve()),
            'model_config': str(Path(args.config).resolve()),
            'fresh_seed': args.seed,
            'source_tensor_counts': dict(counts),
            'official_prefixes': OFFICIAL_PREFIXES,
            'fresh_prefixes': FRESH_PREFIXES,
        },
        'state_dict': merged,
    }
    torch.save(checkpoint, str(output))
    print('[PASS] P0 saved to:', output)
    print('[PASS] Tensor sources:', dict(counts))
    print('[PASS] All {} model keys have compatible shapes.'.format(len(merged)))
    print('[NEXT] Run verify_P0_and_training.py, then A/B0 forward comparison.')


if __name__ == '__main__':
    main()
