#!/usr/bin/env python3
# Diagnostic checkpoint ONLY. Keeps all P0 tensors except restores the
# decoder and prediction-head tensors from the original PGFR checkpoint.
import argparse
from pathlib import Path
import torch

PREFIXES = ('bbox_head.decoder.', 'bbox_head.prediction_heads.')

def get_sd(ckpt):
    sd = ckpt.get('state_dict', ckpt)
    if not isinstance(sd, dict):
        raise TypeError('Checkpoint state_dict is not a dictionary')
    return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0', required=True)
    ap.add_argument('--pgfr', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    p0_obj = torch.load(a.p0, map_location='cpu', weights_only=False)
    pgfr_obj = torch.load(a.pgfr, map_location='cpu', weights_only=False)
    p0 = get_sd(p0_obj)
    pgfr = get_sd(pgfr_obj)
    keys = [k for k in p0 if k.startswith(PREFIXES)]
    if len(keys) != 84:
        raise RuntimeError(f'Expected exactly 84 official-tail keys, found {len(keys)}')
    for key in keys:
        if key not in pgfr:
            raise KeyError(f'Missing PGFR tensor: {key}')
        if pgfr[key].shape != p0[key].shape:
            raise RuntimeError(f'Shape mismatch for {key}')
        p0[key] = pgfr[key].detach().clone()
    if 'state_dict' in p0_obj:
        p0_obj.setdefault('meta', {})['diagnostic_control'] = (
            'PGFR decoder/prediction_heads restored; all other P0 tensors unchanged')
        output_obj = p0_obj
    else:
        output_obj = p0
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_obj, output)
    print(f'[PASS] Saved diagnostic control checkpoint: {output}')
    print(f'[PASS] Restored {len(keys)} decoder/head tensors from PGFR')
    print('[NOTE] This is a diagnostic control, NOT the agreed Official-tail P0.')

if __name__ == '__main__':
    main()
