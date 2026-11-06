#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify provenance of PGFR/Official hybrid P0 and optionally freeze audit B-E.

P0 verification requires PGFR and Official checkpoints.
To audit a trained B/C/D/E checkpoint, additionally pass --trained and --variant.
The script cannot by itself prove real forward behavior or training quality.
"""

import argparse
from collections import Counter
import sys
import torch

OFFICIAL = ('bbox_head.decoder.', 'bbox_head.prediction_heads.')
FRESH = (
    'feat_projector.',
    'bbox_head.fusion_cross_attention.', 'bbox_head.fusion_ffn.',
    'bbox_head.fusion_norm1.', 'bbox_head.fusion_norm2.',
    'bbox_head.camera_proposal_pos_embedding.',
    'bbox_head.bev_query_pos_embedding.',
    'bbox_head.refined_attention.', 'bbox_head.refined_ffn.',
    'bbox_head.refined_norm1.', 'bbox_head.refined_norm2.',
    'bbox_head.rrrf_query_selector.',
    'bbox_head.calibration_predictor.',
    'bbox_head.residual_', 'bbox_head.refined_fusion_aux_head.',
)
COARSE = (
    'feat_projector.',
    'bbox_head.fusion_cross_attention.', 'bbox_head.fusion_ffn.',
    'bbox_head.fusion_norm1.', 'bbox_head.fusion_norm2.',
    'bbox_head.camera_proposal_pos_embedding.',
    'bbox_head.bev_query_pos_embedding.',
)
REFINED = (
    'bbox_head.refined_attention.', 'bbox_head.refined_ffn.',
    'bbox_head.refined_norm1.', 'bbox_head.refined_norm2.',
)
LOCAL = ('bbox_head.rrrf_query_selector.',)
GROUPS = {
    'decoder': ('bbox_head.decoder.',),
    'prediction_heads': ('bbox_head.prediction_heads.',),
    'coarse': COARSE,
    'refined': REFINED,
    'local': LOCAL,
}
TRAIN_GROUPS = {
    'B': ('decoder', 'prediction_heads'),
    'C': ('decoder', 'prediction_heads', 'coarse'),
    'D': ('decoder', 'prediction_heads', 'coarse', 'refined'),
    'E': ('decoder', 'prediction_heads', 'coarse', 'refined', 'local'),
}
REQUIRED = (
    'corr.', 'z_estimator.', 'calib_head.',
    'bbox_head.shared_conv.', 'bbox_head.heatmap_head.',
    'bbox_head.class_encoding.',
    'bbox_head.decoder.', 'bbox_head.prediction_heads.',
    'feat_projector.', 'bbox_head.fusion_cross_attention.',
    'bbox_head.refined_attention.', 'bbox_head.rrrf_query_selector.',
)


def load_checkpoint(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    sd = obj.get('state_dict', obj)
    norm = {}
    for name, value in sd.items():
        name = name[7:] if name.startswith('module.') else name
        if name in norm:
            raise ValueError('Duplicate normalized key: ' + name)
        norm[name] = value
    return norm, obj.get('meta', {})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p0', required=True)
    ap.add_argument('--pgfr', required=True)
    ap.add_argument('--official', required=True)
    ap.add_argument('--trained', help='Optional B-E trained checkpoint')
    ap.add_argument('--variant', choices=('B', 'C', 'D', 'E'))
    args = ap.parse_args()
    if bool(args.trained) != bool(args.variant):
        ap.error('--trained and --variant must be supplied together')

    p0, meta = load_checkpoint(args.p0)
    pgfr, _ = load_checkpoint(args.pgfr)
    official, _ = load_checkpoint(args.official)
    issues = []
    counts = Counter()

    for prefix in REQUIRED:
        if not any(k.startswith(prefix) for k in p0):
            issues.append('MISSING P0 PREFIX: ' + prefix)

    for key, p0_tensor in p0.items():
        if key.startswith(OFFICIAL):
            source, name = official, 'Official'
        elif key.startswith(FRESH):
            counts['Fresh'] += 1
            continue  # No external donor to compare against.
        else:
            source, name = pgfr, 'PGFR'

        counts[name] += 1
        if key not in source:
            issues.append('MISSING {} SOURCE KEY: {}'.format(name, key))
        elif not torch.equal(p0_tensor, source[key]):
            issues.append('NOT EQUAL {}: {}'.format(name, key))

    print('P0 source tensor counts:', dict(counts))
    print('P0 build seed in metadata:', meta.get('fresh_seed', '<not provided>'))
    if counts['Official'] != 84:
        issues.append('Expected 84 Official decoder/head tensors; observed {}'
                      .format(counts['Official']))
    if counts['Fresh'] == 0:
        issues.append('No freshly initialized RRRF tensors found in P0')

    if args.trained:
        trained, _ = load_checkpoint(args.trained)
        permitted_groups = TRAIN_GROUPS[args.variant]
        changed = Counter()
        changed_tensors = 0
        for key, original in p0.items():
            if key not in trained:
                # The Local selector is instantiated only in E; B/C/D may
                # legitimately omit its keys when loaded with strict=False.
                if args.variant != 'E' and key.startswith(LOCAL):
                    continue
                issues.append('MISSING TRAINED KEY: ' + key)
                continue
            target = trained[key]
            if original.shape != target.shape:
                issues.append('TRAINED SHAPE MISMATCH: ' + key)
                continue
            if torch.equal(original, target):
                continue

            changed_tensors += 1
            matched = [g for g in permitted_groups
                       if key.startswith(GROUPS[g])]
            if not matched:
                issues.append('FROZEN TENSOR CHANGED: ' + key)
            else:
                changed[matched[0]] += 1

        for key in trained:
            if key not in p0:
                issues.append('UNEXPECTED TRAINED KEY: ' + key)

        print('Trainable groups expected for {}: {}'.format(
            args.variant, ', '.join(permitted_groups)))
        print('Changed tensor count:', changed_tensors)
        for group in permitted_groups:
            print('  {}: {} changed tensors'.format(group, changed[group]))
            if not changed[group]:
                issues.append('NO WEIGHT/BUFFER UPDATED IN GROUP: ' + group)

    if issues:
        for issue in issues[:50]:
            print('[ERROR]', issue)
        sys.exit('[FAIL] {} issue(s) found.'.format(len(issues)))

    print('[PASS] P0 provenance validated.' if not args.trained else
          '[PASS] P0 provenance and trained checkpoint freeze audit validated.')
    print('[NOTE] Forward correctness, actual gradients and NDS require GPU tests.')


if __name__ == '__main__':
    main()
