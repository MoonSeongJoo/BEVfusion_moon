#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build P4: P3 (all PGFR) + Official BEVFusion Prediction Heads only.

P4 tensor provenance:
  - Original P3 checkpoint: EVERY tensor except bbox_head.prediction_heads.*
    (including LGPC, RRRF coarse/refined/local, feature encoders, BEV/query,
     Transformer decoder, and all spconv kernels/buffers).
  - Official BEVFusion: bbox_head.prediction_heads.* ONLY.
  - Newly initialized weights: NONE.

This script does not change model architecture, inference configuration,
optimizer state, or model source code. It only writes a merged .pth file.

Run from /workspace/mmdetection3d:
  python tools/rrrf_ablation/build_PGFR_Ablation_P4.py \\
    --p3 data/weights/ablation_v2/PGFR_Ablation_P3.pth \\
    --official data/weights/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \\
    --output data/weights/ablation_v2/PGFR_Ablation_P4.pth

IMPORTANT: Only load checkpoints from trusted sources. torch.load with
weights_only=False may execute code during pickle deserialization.
"""

import argparse
import os
import tempfile
from pathlib import Path

import torch

HEAD_PREFIX = 'bbox_head.prediction_heads.'
DECODER_PREFIX = 'bbox_head.decoder.'


def checkpoint_state(checkpoint, label):
    if not isinstance(checkpoint, dict):
        raise TypeError(f'{label}: checkpoint must be a dictionary')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise TypeError(f'{label}: missing/empty state_dict')
    return state


def index_state(state, label):
    """Normalize optional DDP 'module.' prefix without renaming output keys."""
    index = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            raise TypeError(f'{label}: encountered non-string key {raw_key!r}')
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{label}: non-Tensor value at key {raw_key!r}')
        key = raw_key.removeprefix('module.')
        if key in index:
            raise RuntimeError(f'{label}: duplicate normalized key {key}')
        index[key] = (raw_key, value)
    return index


def verify_head_mapping(p3_index, official_index, expected_count):
    p3_keys = sorted(k for k in p3_index if k.startswith(HEAD_PREFIX))
    official_keys = sorted(k for k in official_index if k.startswith(HEAD_PREFIX))
    decoder_keys = sorted(k for k in p3_index if k.startswith(DECODER_PREFIX))
    if not p3_keys:
        raise RuntimeError(f'P3 contains no {HEAD_PREFIX} tensors')
    if not decoder_keys:
        raise RuntimeError(f'P3 contains no {DECODER_PREFIX} tensors')
    if not official_keys:
        raise RuntimeError(f'Official checkpoint contains no {HEAD_PREFIX} tensors')
    if expected_count > 0 and len(p3_keys) != expected_count:
        raise RuntimeError(
            f'Expected {expected_count} P3 Prediction Head tensors, got {len(p3_keys)}. '
            'Use --expected-head-tensors 0 only after inspecting the architecture.')

    missing = sorted(set(p3_keys) - set(official_keys))
    extra = sorted(set(official_keys) - set(p3_keys))
    if missing or extra:
        raise RuntimeError(
            f'Official/P3 Prediction Head key mismatch: '
            f'missing_in_official={len(missing)}, extra_in_official={len(extra)}\n'
            f'First missing keys: {missing[:12]}\nFirst extra keys: {extra[:12]}')

    mismatches = []
    changed = 0
    for key in p3_keys:
        p3_tensor = p3_index[key][1]
        off_tensor = official_index[key][1]
        if p3_tensor.shape != off_tensor.shape or p3_tensor.dtype != off_tensor.dtype:
            mismatches.append(
                f'{key}: P3={tuple(p3_tensor.shape)} {p3_tensor.dtype}, '
                f'Official={tuple(off_tensor.shape)} {off_tensor.dtype}')
        elif not torch.equal(p3_tensor, off_tensor):
            changed += 1
    if mismatches:
        raise RuntimeError('Prediction Head tensor shape/dtype mismatch:\n' +
                           '\n'.join(mismatches[:20]))
    return p3_keys, decoder_keys, changed


def verify_output(p3_state, p3_index, official_index, p4_state, head_keys):
    """Require that ONLY Prediction Head tensors differ from P3."""
    if set(p3_state) != set(p4_state):
        raise AssertionError('State dict keys changed during P4 merge')
    updated = set()
    for normalized_key, (raw_key, old_tensor) in p3_index.items():
        new_tensor = p4_state[raw_key]
        if normalized_key.startswith(HEAD_PREFIX):
            donor_tensor = official_index[normalized_key][1]
            if not torch.equal(new_tensor, donor_tensor):
                raise AssertionError('Official Prediction Head copy failed: ' + normalized_key)
            updated.add(normalized_key)
        else:
            # Strict reference identity ensures ALL non-head P3 tensors are
            # preserved exactly, including 5D spconv tensors and BN buffers.
            if new_tensor is not old_tensor:
                raise AssertionError('Non-head P3 tensor changed: ' + normalized_key)
    if updated != set(head_keys):
        raise AssertionError('Not every Prediction Head was replaced')
    return len(p3_state) - len(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--p3', required=True,
                        help='Baseline P3 checkpoint, all PGFR weights')
    parser.add_argument('--official', required=True,
                        help='Official BEVFusion pretrained checkpoint')
    parser.add_argument('--output', required=True,
                        help='P4 destination .pth (not allowed to overwrite donors)')
    parser.add_argument('--expected-head-tensors', type=int, default=48,
                        help='Expected bbox_head.prediction_heads tensor count; '
                             '0 disables fixed-count check (default 48)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check compatibility and print report without writing P4')
    args = parser.parse_args()
    if args.expected_head_tensors < 0:
        parser.error('--expected-head-tensors must be >= 0')

    p3_path = Path(args.p3).expanduser().resolve()
    off_path = Path(args.official).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    for path in (p3_path, off_path):
        if not path.is_file():
            parser.error(f'Input checkpoint not found: {path}')
    if p3_path == off_path:
        parser.error('P3 and Official must be separate checkpoints')
    if out_path in (p3_path, off_path):
        parser.error('Never overwrite a donor checkpoint')
    if out_path.suffix.lower() != '.pth':
        parser.error('Output must end in .pth')
    if out_path.exists():
        parser.error(f'Output already exists; refusing to overwrite: {out_path}')

    print(f'[LOAD] P3:       {p3_path}', flush=True)
    p3_obj = torch.load(str(p3_path), map_location='cpu', weights_only=False)
    print(f'[LOAD] Official: {off_path}', flush=True)
    off_obj = torch.load(str(off_path), map_location='cpu', weights_only=False)

    p3_state = checkpoint_state(p3_obj, 'P3')
    off_state = checkpoint_state(off_obj, 'Official')
    p3_index = index_state(p3_state, 'P3')
    off_index = index_state(off_state, 'Official')

    head_keys, decoder_keys, changed = verify_head_mapping(
        p3_index, off_index, args.expected_head_tensors)

    replacements = {}
    for key in head_keys:
        raw_p3_key, _ = p3_index[key]
        donor_tensor = off_index[key][1]
        replacements[raw_p3_key] = donor_tensor.detach().cpu().clone()

    p4_state = dict(p3_state)
    p4_state.update(replacements)
    untouched_count = verify_output(p3_state, p3_index, off_index, p4_state, head_keys)

    print(f'[PASS] Official Prediction Head tensors: {len(head_keys)}/{len(head_keys)}')
    print(f'[PASS] PGFR Decoder tensors preserved: {len(decoder_keys)}/{len(decoder_keys)}')
    print(f'[PASS] All other P3 tensors preserved: {untouched_count}')
    print(f'[CHECK] Head tensors that differ from P3: {changed}/{len(head_keys)}')
    print('[PASS] No random initialization, no training, no spconv layout modification')
    if changed == 0:
        print('[WARNING] Official and P3 Prediction Heads are identical; '
              'P4 is not an informative swap.')
    if args.dry_run:
        print('[DRY RUN] Compatibility verified, checkpoint not saved.')
        return

    # Output is for weight loading ('load_from'), not resuming optimizer states.
    # Avoid copying stale optimizer/parameter scheduler data from original P3.
    if 'state_dict' in p3_obj:
        meta = dict(p3_obj.get('meta', {}) or {})
        previous_sources = dict(meta.get('source_tensor_counts', {}) or {})
        meta.update({
            'purpose': 'P4: PGFR LGPC/RRRF/Decoder + Official BEVFusion Prediction Heads',
            'base_checkpoint': str(p3_path),
            'official_head_donor': str(off_path),
            'decoder_source': 'PGFR (P3 unchanged)',
            'prediction_heads_source': 'Official BEVFusion',
            'official_prefixes': [HEAD_PREFIX],
            'source_tensor_counts_P3_before_swap': previous_sources,
            'source_tensor_counts': {
                'P3_unchanged': untouched_count,
                'Official': len(head_keys),
                'Fresh': 0,
            },
            'prediction_head_replaced_tensor_count': len(head_keys),
            'prediction_head_different_from_P3_count': changed,
            'non_head_tensors_preserved_from_P3': True,
            'strict_loading_validated': False,
            'runtime_forward_validated': False,
            'evaluation_note': 'Use the same PGFR original eval config as P3 '
                               'to compare P3 vs P4; P4 is not Decoupled.',
        })
        output_obj = {'meta': meta, 'state_dict': p4_state}
    else:
        # Plain state_dict is accepted by MMEngine checkpoint loading.
        output_obj = p4_state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.stem + '.tmp_', suffix='.pth', dir=out_path.parent)
    os.close(fd)
    try:
        torch.save(output_obj, tmp_path)
        if os.path.getsize(tmp_path) == 0:
            raise IOError('torch.save produced an empty checkpoint')
        if out_path.exists():
            raise FileExistsError(f'Refusing to replace existing file: {out_path}')
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f'[PASS] P4 saved: {out_path}')
    print('[NEXT] Test with projects/BEVFusion/configs/pgfr_feature_refine_local_eval.py')
    print('[NOTE] Run-time strict model load and NDS are not verified by this script.')


if __name__ == '__main__':
    main()