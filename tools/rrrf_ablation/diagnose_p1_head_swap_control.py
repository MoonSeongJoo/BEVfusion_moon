#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create P3 from P2 by restoring ONLY the PGFR Transformer Decoder.

Experiment:
    P2: PGFR LGPC/RRRF + Official Decoder + PGFR Prediction Heads
    P3: PGFR LGPC/RRRF + PGFR Decoder     + PGFR Prediction Heads

All other P2 tensors (including the spconv checkpoint load layout) remain
untouched. The *same* P2 evaluation config must be used for the P3 test.

Example:
    python tools/rrrf_ablation/diagnose_p1_head_swap_control.py \\
      --p1 data/weights/ablation_v2/PGFR_Ablation_P2.pth \\
      --pgfr data/weights/PGFR_Det_final_model.pth \\
      --output data/weights/ablation_v2/PGFR_Ablation_P3.pth

Only load checkpoints from trusted sources: torch.load can unpickle objects.
No model construction, training, or source code modification is performed.
"""
import argparse
import os
from pathlib import Path
import tempfile

import torch

DECODER_PREFIX = 'bbox_head.decoder.'
HEAD_PREFIX = 'bbox_head.prediction_heads.'


def checkpoint_state(obj, filename):
    if not isinstance(obj, dict):
        raise TypeError(f'{filename}: expected a checkpoint dictionary')
    state = obj.get('state_dict', obj)
    if not isinstance(state, dict):
        raise TypeError(f'{filename}: invalid state_dict')
    return state


def normalized_keys(state, label):
    """Return normalized-name -> (original-name, tensor) mapping."""
    result = {}
    for raw_key, tensor in state.items():
        if not isinstance(raw_key, str):
            raise TypeError(f'{label}: non-string state_dict key')
        key = raw_key.removeprefix('module.')
        if key in result:
            raise RuntimeError(f'{label}: duplicate normalized key {key}')
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{label}: non-tensor entry {raw_key}')
        result[key] = (raw_key, tensor)
    return result


def compare_tensor_shapes(a, b, key):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(
            f'Incompatible tensor {key}: P2 {tuple(a.shape)} {a.dtype}, '
            f'PGFR {tuple(b.shape)} {b.dtype}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # --p1 is retained to support the exact command from the conversation.
    parser.add_argument('--p1', '--p2', dest='p2', required=True,
                        help='P2 checkpoint (historical --p1 name also accepted)')
    parser.add_argument('--pgfr', required=True,
                        help='Original PGFR_Det_final_model.pth checkpoint')
    parser.add_argument('--output', required=True,
                        help='Destination PGFR_Ablation_P3.pth')
    parser.add_argument('--expected-decoder-tensors', type=int, default=36,
                        help='Expected Decoder tensors (default: 36; 0 disables count check)')
    args = parser.parse_args()

    p2_path = Path(args.p2).expanduser().resolve()
    pgfr_path = Path(args.pgfr).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    for src in (p2_path, pgfr_path):
        if not src.is_file():
            parser.error(f'Input checkpoint not found: {src}')
    if out_path in (p2_path, pgfr_path):
        parser.error('Output must not overwrite P2 or PGFR input')
    if out_path.suffix != '.pth':
        parser.error('Output should have a .pth extension')

    print('[LOAD] P2:', p2_path, flush=True)
    # weights_only=False is intentional for own trusted MMEngine checkpoint files.
    p2_obj = torch.load(str(p2_path), map_location='cpu', weights_only=False)
    print('[LOAD] PGFR:', pgfr_path, flush=True)
    pgfr_obj = torch.load(str(pgfr_path), map_location='cpu', weights_only=False)
    p2_state = checkpoint_state(p2_obj, 'P2')
    p2 = normalized_keys(p2_state, 'P2')
    pgfr = normalized_keys(checkpoint_state(pgfr_obj, 'PGFR'), 'PGFR')

    decoder_keys = sorted(k for k in p2 if k.startswith(DECODER_PREFIX))
    head_keys = sorted(k for k in p2 if k.startswith(HEAD_PREFIX))
    if not decoder_keys:
        raise RuntimeError('P2 checkpoint has no bbox_head.decoder.* tensors')
    if not head_keys:
        raise RuntimeError('P2 checkpoint has no bbox_head.prediction_heads.* tensors')
    if args.expected_decoder_tensors and len(decoder_keys) != args.expected_decoder_tensors:
        raise RuntimeError(f'Expected {args.expected_decoder_tensors} Decoder tensors, '
                           f'but P2 contains {len(decoder_keys)}')

    # If P2 was produced by the supplied P2 builder, its heads must already
    # match the PGFR donor. Validate this before calling this an isolated test.
    for key in head_keys:
        if key not in pgfr:
            raise RuntimeError(f'PGFR Prediction Head key missing: {key}')
        p2_value = p2[key][1]
        pgfr_value = pgfr[key][1]
        compare_tensor_shapes(p2_value, pgfr_value, key)
        if not torch.equal(p2_value, pgfr_value):
            raise RuntimeError(
                f'P2 Prediction Head is NOT identical to PGFR: {key}. '
                'Cannot isolate Decoder-only effect using this input.')

    extra_decoder = sorted(k for k in pgfr if k.startswith(DECODER_PREFIX)
                           and k not in p2)
    if extra_decoder:
        raise RuntimeError(f'PGFR has extra Decoder keys not in P2: {extra_decoder[:8]}')

    replacement = {}
    changed = 0
    for key in decoder_keys:
        if key not in pgfr:
            raise RuntimeError(f'PGFR Decoder key missing: {key}')
        raw_key, p2_value = p2[key]
        pgfr_value = pgfr[key][1]
        compare_tensor_shapes(p2_value, pgfr_value, key)
        if not torch.equal(p2_value, pgfr_value):
            changed += 1
        replacement[raw_key] = pgfr_value.detach().cpu().clone()

    # Copy the mapping but preserve all non-Decoder tensor objects as-is.
    p3_state = dict(p2_state)
    p3_state.update(replacement)
    for raw_key, old_tensor in p2_state.items():
        normalized_key = raw_key.removeprefix('module.')
        if normalized_key.startswith(DECODER_PREFIX):
            if not torch.equal(p3_state[raw_key], pgfr[normalized_key][1]):
                raise AssertionError('Decoder donor equality failed: ' + normalized_key)
        else:
            if p3_state[raw_key] is not old_tensor:
                raise AssertionError('Non-Decoder reference unexpectedly changed: ' + raw_key)

    if 'state_dict' in p2_obj:
        p3_obj = dict(p2_obj)
        p3_obj['state_dict'] = p3_state
        meta = dict(p2_obj.get('meta', {}) or {})
        original_counts = dict(meta.get('source_tensor_counts', {}) or {})
        meta.update({
            'purpose': 'P3: PGFR LGPC/RRRF + PGFR Decoder + PGFR Prediction Heads; no training',
            'base_checkpoint': str(p2_path),
            'decoder_donor': str(pgfr_path),
            'decoder_source': 'PGFR',
            'prediction_heads_source': 'PGFR (verified identical)',
            'decoder_replaced_tensor_count': len(decoder_keys),
            'decoder_different_from_P2_count': changed,
            'prediction_heads_verified_tensor_count': len(head_keys),
            'source_tensor_counts_P2_before_swap': original_counts,
            'official_prefixes': (),
            'source_tensor_counts': {'PGFR': len(p3_state), 'Official': 0, 'Fresh': 0},
            'non_decoder_tensors_preserved_from_P2': True,
            'strict_loading_validated_for_P3': False,
            'test_config_instruction': 'Use exactly the P2_eval_config.py, data, and seed',
        })
        # The P2 builder's strict-loading flag is historical and not proof that
        # this newly modified checkpoint has undergone model-level strict load.
        if 'strict_loading_validated' in meta:
            meta['strict_loading_validated'] = False
        p3_obj['meta'] = meta
    else:
        p3_obj = p3_state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic rename: interrupted saves should not corrupt the final destination.
    fd, temp_name = tempfile.mkstemp(prefix=out_path.stem + '.tmp_',
                                     suffix='.pth', dir=out_path.parent)
    os.close(fd)
    try:
        torch.save(p3_obj, temp_name)
        os.replace(temp_name, out_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    print(f'[PASS] Saved P3: {out_path}')
    print(f'[PASS] Decoder tensors restored from PGFR: {len(decoder_keys)} / {len(decoder_keys)}')
    print(f'[CHECK] Decoder tensors different from P2: {changed} / {len(decoder_keys)}')
    print(f'[PASS] Prediction Head tensors already identical to PGFR: {len(head_keys)}')
    print('[PASS] All non-Decoder P2 tensors preserved (including spconv load layout)')
    if changed == 0:
        print('[WARNING] P2 and PGFR Decoder tensors were already identical; '
              'P3 will not provide a meaningful Decoder-swap comparison.')
    print('[NEXT] Evaluate P3 using the same PGFR_Ablation_P2_eval_config.py as P2.')


if __name__ == '__main__':
    main()