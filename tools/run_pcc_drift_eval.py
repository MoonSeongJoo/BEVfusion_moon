#!/usr/bin/env python3
"""Run original OpenMMLab configs with a shared scene-aware drift manifest.

Only evaluation configuration is mutated in memory. Train configs remain intact.
Use --check-only to validate metadata/manifest/config before expensive GPU runs.
"""
import argparse
import os
import pickle
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from projects.drift_eval.protocol import DriftLookup


def leaf(cfg):
    while 'pipeline' not in cfg and 'dataset' in cfg:
        cfg = cfg['dataset']
    if 'pipeline' not in cfg:
        raise RuntimeError('Dataset has no pipeline')
    return cfg


def metadata_tokens(ann_file, data_root):
    p = Path(ann_file)
    if not p.is_absolute():
        p = Path(data_root) / p
    if not p.is_file():
        raise FileNotFoundError('Validation metadata not found: %s' % p)
    with p.open('rb') as f:
        obj=pickle.load(f)
    rows=(obj.get('data_list', obj.get('infos')) if isinstance(obj, dict) else obj)
    if rows is None:
        raise RuntimeError('Unknown nuScenes info pickle layout: '+str(p))
    tokens=set()
    for r in rows:
        t=r.get('token') or r.get('sample_token')
        if not isinstance(t,str) or len(t)<20:
            raise RuntimeError('Missing nuScenes token in '+str(p))
        tokens.add(t)
    if len(tokens)!=len(rows):
        raise RuntimeError('Duplicate sample tokens in '+str(p))
    return tokens,p


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('config')
    pa.add_argument('checkpoint', nargs='?', default=None)
    pa.add_argument('--drift-manifest',required=True)
    pa.add_argument('--mode',choices=['pcc','official','broken','oracle','lccnet'],required=True)
    pa.add_argument('--work-dir',default='work_dirs/drift_evaluation')
    pa.add_argument('--check-only',action='store_true')
    pa.add_argument('--workers',type=int,default=0)
    args=pa.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    # Single GPU / torchrun DDP 자동 구분
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        cfg.launcher = 'pytorch'
    else:
        cfg.launcher = 'none'
    dataset=leaf(cfg.test_dataloader.dataset)
    lookup=DriftLookup(args.drift_manifest)
    expected,ann_path=metadata_tokens(dataset['ann_file'],dataset.get('data_root',''))
    missing=expected-set(lookup.samples)
    if missing:
        raise RuntimeError(f'Manifest missing {len(missing)} validation tokens; first={sorted(missing)[0]}')
    print(f'[COVERAGE] {len(expected)} validation tokens, manifest {len(lookup.samples)} samples, '
          f'scenario={lookup.manifest["scenario"]}, seed={lookup.manifest["seed"]}, '
          f'file={ann_path}')
    cfg.test_dataloader['num_workers']=args.workers
    cfg.test_dataloader['persistent_workers']=bool(args.workers)
    if args.mode=='pcc':
        cfg.model['calibration_mode']='pcc_full'
        transforms=[t for t in dataset['pipeline'] if t['type']=='PointToMultiViewDepth']
        if len(transforms)!=1:
            raise RuntimeError(f'Expected one PointToMultiViewDepth, found {len(transforms)}')
        transforms[0]['drift_manifest']=args.drift_manifest
        # Keep test transform and original evaluation config unchanged otherwise.
    else:
        cfg.model['calibration_mode']=args.mode
        cfg.model['drift_manifest']=args.drift_manifest
        packs=[t for t in dataset['pipeline'] if t['type']=='Pack3DDetInputs']
        if len(packs)!=1:
            raise RuntimeError(f'Expected one Pack3DDetInputs, found {len(packs)}')
        meta=list(packs[0]['meta_keys'])
        for key in ('token','camera_names'):
            if key not in meta:
                meta.append(key)
        packs[0]['meta_keys']=meta
    print('[CONFIG] model=',args.mode,'drift=',args.drift_manifest)
    print('[CONFIG] meta keys patched=' if args.mode!='pcc' else '[CONFIG] PCC transform patched')
    print('[CONFIG] evaluator=',cfg.get('test_evaluator'))
    if args.check_only:
        print('[PASS] Config and manifest match. No model was run.')
        return
    if not args.checkpoint:
        pa.error('checkpoint is required unless --check-only')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError('Checkpoint not found: '+args.checkpoint)
    cfg.load_from=args.checkpoint
    Path(args.work_dir).mkdir(parents=True,exist_ok=True)
    # 두 GPU가 동시에 같은 Config 파일을 쓰지 않도록 Rank 0만 저장
    if int(os.environ.get('RANK', '0')) == 0:
        cfg.dump(
            str(Path(args.work_dir) / 'resolved_drift_config.py')
        )
    Runner.from_cfg(cfg).test()


if __name__=='__main__':
    main()
