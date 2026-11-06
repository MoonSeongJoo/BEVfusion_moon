#!/usr/bin/env python3
"""Summarize a nuScenes submission JSON exported from NuScenesMetric.format_results."""
import argparse
import collections
import json
import math
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('json_path')
    args = p.parse_args()
    with open(args.json_path, encoding='utf-8') as f:
        obj = json.load(f)
    samples = obj.get('results')
    if not isinstance(samples, dict):
        raise ValueError('Expected nuScenes submission JSON with a results dictionary')
    boxes = [b for annos in samples.values() for b in annos]
    score = [float(b.get('detection_score', 0.0)) for b in boxes]
    cls = collections.Counter(b.get('detection_name', '<missing>') for b in boxes)
    nonempty = sum(bool(v) for v in samples.values())
    print('[NUSCENE SUBMISSION AUDIT]')
    print('sample tokens:', len(samples))
    print('nonempty samples:', nonempty)
    print('submitted boxes:', len(boxes))
    print('class counts:', dict(cls))
    if score:
        print('score min / median / max:', min(score), statistics.median(score), max(score))
        print('scores <= 0.01:', sum(s <= .01 for s in score))
        print('first sample annotations:', boxes[:2])
        bad = sum(not all(math.isfinite(float(v)) for v in b.get('translation', [])) for b in boxes)
        print('non-finite center boxes:', bad)
    if len(boxes) == 0:
        print('[WARNING] No boxes reached nuScenes evaluation. Inspect raw decode and score filtering.')
    else:
        print('[NOTE] Boxes exist. If AP=0, examine sample-token mapping and box coordinates/classes.')

if __name__ == '__main__':
    main()
