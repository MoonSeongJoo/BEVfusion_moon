#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-frame, same-perturbation qualitative comparison for the PCC / nuScenes fork.

Two model types register as ``BEVFusion``. Never import both in one process.
The orchestrator runs two FRESH Python subprocesses (PCC then Official+LCCNet),
compares their frame/calibration identity, and renders GT / LCCNet+Official / PCC
in three columns with the exact same original camera photo.

Outputs:
  <stem>_01_lccnet_official.png
  <stem>_02_pcc.png
  <stem>_comparison_2col.png
  <stem>_detection_only_2col.png
  <stem>_GT_only.png
  <stem>_detection_only_3col_GT_LCCNet_PCC.png       [NEW, main paper figure]
  <stem>_comparison_3col_GT_LCCNet_PCC.png           [NEW, overlay + detection]
  <stem>_GT_3D_boxes.png                            [NEW]
  <stem>_manifest.json
  optionally <stem>_comparison_2col_bev.png (with --with-bev)
  --auto-search-count N: run N paired comparisons, class-aware GT match, count-focused
       ranking and selected_top_k/ 3-column examples (with cache reuse).
  --auto-rank-only: recompute rank/figures from existing verified per-frame caches,
       WITHOUT rerunning expensive neural inference.

Frame-selection protocol (qualitative examples, not nuScenes official metrics):
 * Visible GT box count >= 3
 * PCC visible prediction count differs from GT count by <= 1
 * LCCNet visible prediction count differs from GT count by >= 2
 * PCC class-aware GT matches >= LCCNet matches + 1
 * PCC unmatched predictions <= 2, and excess unmatched vs LCCNet <= 1
 * Rank: 3 * extra PCC matches + 2 * extra LCCNet count deviation
         - 1.5 * max(excess PCC unmatched predictions, 0)
 * Selection is explicitly favorable-case selection; overall NDS/mAP must be
   reported separately on full validation data.


Core guarantees:
 * Camera background: same ORIGINAL jpg pixels, not augmented training image.
 * 3D boxes: direct prediction in original LiDAR coordinates, projected with
   CLEAN camera extrinsics (same coordinate system for both detectors).
 * Calibration visualization: exact same LiDAR returns, same GT and same broken
   per-camera SE(3). LCCNet corrected transform is intercepted from the
   official model; PCC Stage-1 corrected transform is reconstructed from model
   prediction meta. RRRF refines camera proposals; its residual is NOT
   presented as a second complete camera projection.
 * If official nuScenes info uses different frames, fail rather than silently
   claiming a matched comparison.
 * GT is read from the nuScenes official annotation API at the matched sample
   token, in the ORIGINAL camera coordinate system (not a prediction cache).
 * Only the ten nuScenes detection classes are drawn; unlike prediction boxes,
   GT boxes have no confidence score and are not thresholded.
 * Matching yields DIAGNOSTIC TP/FP/FN proxies, NOT official nuScenes TP/FP/FN, mAP or NDS.

Requirements: Run from repository root with the model dependencies installed.
"""

import argparse
import csv
import copy
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
GT_COLOR = (60, 220, 50)            # BGR green
BROKEN_COLOR = (55, 60, 240)        # BGR red
RECOVER_COLOR = (0, 215, 255)       # BGR yellow
OFFICIAL_BOX_COLOR = (255, 215, 45) # BGR cyan
PCC_BOX_COLOR = (65, 225, 75)      # BGR green
GT_BOX_COLOR = (10, 165, 255)      # BGR orange; distinct from prediction colors
BOX_EDGES = ((0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--pcc-config', default='projects/BEVFusion/configs/pcc_bevfusion_rrrf_exp4_joint_fresh_eval.py')
    p.add_argument('--pcc-checkpoint', default='work_dirs/pcc_exp4_joint_fresh_400k/iter_83000.pth')
    p.add_argument('--official-config', default='projects/OfficialBEVFusion/configs/bevfusion_v110_lccnet.py')
    p.add_argument('--official-checkpoint', required=True,
                   help='Official BEVFusion detector weights, not LCCNet calibration best_joint_model.pth')
    p.add_argument('--lccnet-model-checkpoint', default='',
                   help='Override model.lccnet_checkpoint within Official config if desired')
    p.add_argument('--sample-idx', type=int, default=17,
                   help='PCC dataset index. The Official index is searched by LiDAR frame identity.')
    p.add_argument('--official-sample-idx', type=int, default=-1,
                   help='Optional matched Official dataset index if auto lookup fails; still validated')
    p.add_argument('--cam-name', choices=CAMERAS, default='CAM_FRONT')
    p.add_argument('--out-dir', default='work_dirs/paper_vis_detection_compare')
    p.add_argument('--score-thr', type=float, default=0.25)
    p.add_argument('--max-boxes', type=int, default=30)
    p.add_argument('--max-points', type=int, default=2600)
    p.add_argument('--min-range', type=float, default=2.0)
    p.add_argument('--max-range', type=float, default=65.0)
    p.add_argument('--max-depth', type=float, default=70.0)
    p.add_argument('--point-radius', type=int, default=1)
    # Paper-only visualization; it does not change prediction or geometry.
    # Enabled automatically when sample_idx == 25.
    p.add_argument('--sample25-no-enhance', action='store_true',
                   help='Disable sample-25 special projection-point drawing.')
    p.add_argument('--sample25-stride', type=int, default=2,
                   help='For sample 25 only: draw every N-th matched LiDAR point.')
    p.add_argument('--sample25-core-radius', type=int, default=2,
                   help='For sample 25 only: colored dot radius (pixels).')
    p.add_argument('--sample25-outline-radius', type=int, default=3,
                   help='For sample 25 only: dark-outline radius (pixels).')
    p.add_argument('--sample25-halo-sigma', type=float, default=3.2,
                   help='For sample 25 only: soft glow Gaussian sigma in pixels.')
    p.add_argument('--sample25-glow-alpha', type=float, default=0.72,
                   help='For sample 25 only: glow intensity in [0,1].')
    p.add_argument('--scale', type=int, default=1)
    p.add_argument('--with-bev', action='store_true')
    p.add_argument('--reuse-worker-cache', action='store_true',
                   help='Skip neural inference and compose figures from matching '
                        'pcc.npz/official.npz and worker metadata in --out-dir')
    p.add_argument('--nuscenes-root', default='data/nuscenes',
                   help='nuScenes dataroot containing v1.0-trainval/ and samples/')
    p.add_argument('--nuscenes-version', default='v1.0-trainval',
                   help='Version containing GT sample_annotation (not v1.0-test)')
    p.add_argument('--paper-crop', type=int, nargs=4, default=None,
                   metavar=('X0','Y0','X1','Y1'),
                   help='Optional common original-image ROI applied equally to all 3 panels')
    p.add_argument('--seed', type=int, default=20260811)
    p.add_argument('--device', default='cuda:0')
    # Auto-select qualitative examples using GT matching. A positive count
    # switches from single-sample rendering to the scan/rank/top-K workflow.
    p.add_argument('--auto-search-count', type=int, default=0,
                   help='Scan this many PCC validation samples; 0 = single-sample mode.')
    p.add_argument('--auto-search-start', type=int, default=0,
                   help='First PCC validation index in automatic search.')
    p.add_argument('--auto-search-step', type=int, default=1)
    p.add_argument('--auto-top-k', type=int, default=3)
    p.add_argument('--auto-match-distance', type=float, default=2.0,
                   help='Ground-plane center matching distance in meters (qualitative proxy only).')
    p.add_argument('--auto-min-visible-gt', type=int, default=3,
                   help='Minimum camera-visible nuScenes GT objects before a frame can qualify.')
    p.add_argument('--auto-min-gain', type=int, default=1,
                   help='Minimum GT objects matched by PCC but NOT by LCCNet.')
    p.add_argument('--auto-max-extra-fp', type=int, default=1,
                   help='Maximum excess unmatched PCC predictions relative to LCCNet.')
    # Count-focused paper figure selection. All box counts refer to the SAME
    # camera-visible 10 nuScenes detection classes, not all 360-degree LiDAR boxes.
    p.add_argument('--auto-max-ours-count-gap', type=int, default=1,
                   help='Require abs(PCC visible boxes - GT visible boxes) <= this value.')
    p.add_argument('--auto-min-lcc-count-gap', type=int, default=2,
                   help='Require abs(LCC visible boxes - GT visible boxes) >= this value.')
    p.add_argument('--auto-min-net-tp-gain', type=int, default=1,
                   help='Require PCC proxy TP count >= LCC proxy TP count + this value.')
    p.add_argument('--auto-max-ours-fp', type=int, default=2,
                   help='Require PCC unmatched prediction count <= this value.')
    p.add_argument('--auto-lcc-under-detect', action='store_true',
                   help='Require LCC visible boxes <= GT visible boxes - min LCC count gap.')
    p.add_argument('--auto-rank-only', action='store_true',
                   help='Scan existing verified per-frame cached predictions, skip ALL inference; '
                        'use for re-ranking previously scanned samples after changing selection rules.')
    p.add_argument('--auto-allow-no-gain', action='store_true',
                   help='Permit rendering top frames even when no improvement qualifies; results clearly flagged.')
    p.add_argument('--auto-reuse-cache', action='store_true',
                   help='Reuse per-frame cached workers only after checkpoint/identity validation.')
    p.add_argument('--auto-force-rerun', action='store_true',
                   help='Recompute per-frame predictions even when a valid cache exists.')
    p.add_argument('--auto-skip-errors', action='store_true',
                   help='Record individual failures in CSV and continue (default aborts).')
    p.add_argument('--run-worker', choices=('pcc','official'), default='',
                   help=argparse.SUPPRESS)
    return p.parse_args()


def verify_exists(path, what):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{what}: {path}')


def resolve_path(path):
    if not path:
        raise RuntimeError('Missing image/lidar path in metainfo')
    f = Path(str(path))
    choices = (f, Path.cwd()/f, Path.cwd()/'data/nuscenes'/f,
               Path('/workspace/mmdetection3d')/f,
               Path('/workspace/mmdetection3d/data/nuscenes')/f)
    for a in choices:
        if a.is_file(): return str(a.resolve())
    raise FileNotFoundError(f'Cannot resolve input file {f}; attempted {[str(c) for c in choices]}')


def paths_in_meta(meta, key):
    obj = meta.get(key)
    if isinstance(obj, (list,tuple)): return [str(x) for x in obj]
    if isinstance(obj, dict):
        if all(cam in obj for cam in CAMERAS): return [str(obj[cam]) for cam in CAMERAS]
    raise RuntimeError(f'Expected {key} list of six camera paths; received {type(obj)}')


def camera_order_check(paths):
    if len(paths)!=6: raise RuntimeError(f'Expected 6 cameras, got {len(paths)}')
    for i,(p,cam) in enumerate(zip(paths,CAMERAS)):
        # Don't accidentally confuse CAM_FRONT and CAM_FRONT_LEFT.
        parts = set(Path(p).parts)
        if any(t.startswith('CAM_') for t in parts) and cam not in parts:
            raise RuntimeError(f'Unexpected camera order at index={i}: {p}, wanted {cam}')


def image_key(path):
    return Path(str(path)).name


def float_matrix(arr, shape):
    a = np.asarray(arr, dtype=np.float64)
    if a.shape != shape: raise RuntimeError(f'Expected matrix shape={shape}, got {a.shape}')
    if not np.isfinite(a).all(): raise RuntimeError('Matrix has NaNs/Infs')
    return a


def make_runner(config_file, checkpoint, device, override_lccnet=''):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    cfg=Config.fromfile(config_file)
    cfg.launcher='none'; cfg.load_from=None; cfg.resume=False
    cfg.work_dir='work_dirs/_qual_worker_'+('official' if override_lccnet else 'pcc')
    cfg.test_dataloader.batch_size=1
    cfg.test_dataloader.num_workers=0
    cfg.test_dataloader.persistent_workers=False
    cfg.test_dataloader.sampler.shuffle=False
    if override_lccnet:
        cfg.model.lccnet_checkpoint=override_lccnet
    runner=Runner.from_cfg(cfg)
    # The official detector checkpoint can omit LCCNet (separately loaded in model ctor).
    checkpoint_obj=load_checkpoint(runner.model, checkpoint, map_location='cpu')
    if not checkpoint_obj: print('[WARN] Checkpoint loader returned no metadata')
    model=runner.model.to(device).eval()
    if hasattr(model,'lccnet') and model.lccnet is not None:
        model.lccnet.eval()
    return runner,model


def nth_batch(runner, index):
    if index < 0: raise ValueError('Negative sample index')
    for i, raw in enumerate(runner.test_dataloader):
        if i==index: return raw
    raise IndexError(f'Sample {index} not in test dataloader')


def processed_batch(model, raw):
    proc=model.data_preprocessor(raw,training=False)
    assert len(proc['data_samples']) == 1
    return proc['inputs'],proc['data_samples']


def data_sample_path(meta):
    v=meta.get('lidar_path')
    if v: return str(v)
    p=meta.get('lidar_points',{})
    if isinstance(p,dict): return str(p.get('lidar_path',''))
    return ''


def sample_id(meta):
    # The LiDAR file basename is a unique physical-frame identifier in nuScenes.
    lidar=data_sample_path(meta)
    if not lidar: raise RuntimeError('Sample missing lidar_path; cannot pair datasets')
    return image_key(lidar)


def get_boxes(pred):
    import torch
    inst=getattr(pred,'pred_instances_3d',None)
    if inst is None: raise RuntimeError('Missing pred_instances_3d')
    bboxes=getattr(inst,'bboxes_3d',None)
    scores=getattr(inst,'scores_3d',None)
    labels=getattr(inst,'labels_3d',None)
    if bboxes is None or scores is None or labels is None:
        raise RuntimeError('Missing detection bboxes_3d / scores_3d / labels_3d')
    if not hasattr(bboxes,'corners'):
        raise RuntimeError(f'Expected LiDARInstance3DBoxes.corners, got {type(bboxes)}')
    if 'lidar' not in bboxes.__class__.__name__.lower():
        raise RuntimeError('Boxes are not LiDARInstance3DBoxes; must convert to LiDAR before projecting')
    corners=bboxes.corners.detach().cpu().numpy().astype(np.float32)
    bs=bboxes.tensor.detach().cpu().numpy().astype(np.float32)
    sc=scores.detach().cpu().numpy().astype(np.float32)
    la=labels.detach().cpu().numpy().astype(np.int32)
    if corners.shape != (len(sc),8,3): raise RuntimeError('Unexpected LiDAR box corners')
    return dict(boxes=bs,corners=corners,scores=sc,labels=la)


def matrix_dict_from_pcc(sample):
    import torch
    Tclean=torch.as_tensor(getattr(sample,'camera2lidar')).detach().cpu().double().numpy()
    Tbroken=torch.as_tensor(getattr(sample,'broken_camera2lidar')).detach().cpu().double().numpy()
    K=torch.as_tensor(getattr(sample,'broken_camera_intrinsics')).detach().cpu().double().numpy()
    for name,val in (('clean',Tclean),('broken',Tbroken),('K',K)):
        float_matrix(val,(6,4,4))
    return Tclean,Tbroken,K


def compute_pcc_stage1(model,sample,broken,K):
    import torch
    mi=sample.metainfo
    r=mi.get('stage1_pred_delta_rot',mi.get('pred_delta_rot_1st'))
    t=mi.get('stage1_pred_delta_trans',mi.get('pred_delta_trans_1st'))
    if r is None or t is None:
        raise RuntimeError('PCC Stage-1 prediction absent in sample.metainfo')
    dev=next(model.parameters()).device
    rot=torch.as_tensor(r,device=dev,dtype=torch.float32).reshape(1,6,3)
    trans=torch.as_tensor(t,device=dev,dtype=torch.float32).reshape(1,6,3)
    T=torch.as_tensor(broken,device=dev,dtype=torch.float32)[None]
    intrinsic=torch.as_tensor(K,device=dev,dtype=torch.float32)[None]
    out=model._get_corrected_calib_from_prediction(rot,trans,T,intrinsic)['cam2lidar']
    return out[0].detach().cpu().double().numpy()


def pcc_worker(args,dirpath):
    import torch
    runner,model=make_runner(args.pcc_config,args.pcc_checkpoint,args.device)
    raw=nth_batch(runner,args.sample_idx)
    inp,samples=processed_batch(model,raw)
    s=samples[0]; clean,broken,K=matrix_dict_from_pcc(s)
    meta=s.metainfo; paths=paths_in_meta(meta,'img_path'); camera_order_check(paths)
    aug=np.asarray(meta.get('lidar_aug_matrix',np.eye(4)),dtype=np.float64)
    if aug.ndim==3: aug=aug[0]
    if aug.shape != (4,4) or not np.allclose(aug,np.eye(4),atol=2e-4):
        raise RuntimeError('PCC test LiDAR augmentation is not identity; box frame needs explicit undo')
    with torch.no_grad(): result=model(inp,samples,mode='predict')[0]
    boxes=get_boxes(result)
    s1=compute_pcc_stage1(model,result,broken,K)
    dclean={'frame_key':sample_id(meta),'token':str(meta.get('token',meta.get('sample_idx',''))),
            'img_paths':paths,'lidar_path':data_sample_path(meta),
            'dataset_index':args.sample_idx,'camera_names':list(CAMERAS),
            'pcc_config':args.pcc_config,'pcc_checkpoint':args.pcc_checkpoint}
    save_npz(dirpath/'pcc.npz',boxes,clean=clean,broken=broken,K=K,recovered=s1)
    write_json(dirpath/'pcc_meta.json',dclean)
    print('[PCC] frame=',dclean['frame_key'],'boxes=',len(boxes['scores']))


def info_lidar_candidate(info):
    if not isinstance(info,dict): return ''
    if 'lidar_path' in info: return str(info['lidar_path'])
    points=info.get('lidar_points')
    if isinstance(points,dict): return str(points.get('lidar_path',''))
    return ''


def find_official_index(dataset,target,provided):
    if provided>=0:
        indices=(provided,)
    else:
        indices=range(len(dataset))
    for i in indices:
        # get_data_info uses annotation metadata (not full LiDAR loading).
        info=dataset.get_data_info(i)
        lidar=info_lidar_candidate(info)
        if lidar and image_key(lidar)==target:
            return i
        # Some data adapters store the resolved path in a nested dict.
        if not lidar and hasattr(dataset,'data_list') and i<len(dataset.data_list):
            lidar=info_lidar_candidate(dataset.data_list[i])
            if lidar and image_key(lidar)==target: return i
    raise RuntimeError(
        f'Official dataset frame not found: lidar basename={target}. '
        'Official/PCC information files may not overlap. Do not substitute same index silently; '
        'try --official-sample-idx only after confirming its frame.')


def official_worker(args,dirpath):
    import torch
    ref=json.loads((dirpath/'pcc_meta.json').read_text())
    ref_np=np.load(str(dirpath/'pcc.npz'))
    pcc_gt=ref_np['clean']; pcc_br=ref_np['broken']
    delta=pcc_br @ np.linalg.inv(pcc_gt)
    override=args.lccnet_model_checkpoint
    runner,model=make_runner(args.official_config,args.official_checkpoint,args.device,
                             override_lccnet=override)
    if getattr(model,'calibration_mode',None) != 'lccnet':
        raise RuntimeError('Official config must set model.calibration_mode="lccnet"')
    idx=find_official_index(runner.test_dataloader.dataset,ref['frame_key'],args.official_sample_idx)
    raw=nth_batch(runner,idx)
    inp,samples=processed_batch(model,raw)
    s=samples[0]; meta=s.metainfo
    paths=paths_in_meta(meta,'img_path'); camera_order_check(paths)
    if sample_id(meta)!=ref['frame_key']:
        raise RuntimeError('Same-frame gate failed after Official dataloader preprocessing')
    if any(image_key(a)!=image_key(b) for a,b in zip(paths,ref['img_paths'])):
        raise RuntimeError('Camera images differ between PCC and Official; cannot compare')
    orig=np.asarray(meta['cam2lidar'],dtype=np.float64)
    float_matrix(orig,(6,4,4))
    k_official=np.asarray(meta['cam2img'],dtype=np.float64)
    float_matrix(k_official,(6,4,4))
    gt_diff=float(np.max(np.abs(orig-pcc_gt)))
    k_diff=float(np.max(np.abs(k_official-ref_np['K'])))
    print('[PAIR GATE] official-vs-PCC clean c2l max diff=',gt_diff,
          ' camera K max diff=',k_diff)
    if gt_diff > 3e-3 or k_diff > 3e-3:
        raise RuntimeError('Official and PCC clean extrinsic/K matrices differ. '
                           'Cannot force same perturbation until calibration conventions match.')
    # Patch only the *sampled perturbation* of the Official model; the Official
    # LCCNet and BEVFusion detector implementation remains unchanged.
    # B=1 camera sequence unchanged (validated by image path gate).
    observed={'called':0,'corrected':[]}
    original_broken_builder=model._build_broken_camera2lidar
    original_build=model._build_calib_dict_from_cam2lidar

    def shared_perturbation(camera2lidar_gt,batch_input_metas):
        if camera2lidar_gt.shape[0] != 1 or camera2lidar_gt.shape[1] != 6:
            raise RuntimeError('Only B=1, six cameras supported')
        clean=camera2lidar_gt
        d=torch.as_tensor(delta,device=clean.device,dtype=clean.dtype)[None]
        broken=torch.matmul(d,clean)
        observed['called']+=1
        observed['broken']=broken.detach().cpu().numpy()[0]
        # Expected Official method returns (broken_camera2lidar, delta_gt, debug)
        return broken,d,{'origin':'PCC paired perturbation','seed':'not re-sampled'}

    def capture_calib(candidate, camera_intrinsics):
        raw_result=original_build(candidate,camera_intrinsics)
        arr=candidate.detach().cpu().numpy()
        if arr.shape == (1,6,4,4):
            observed['corrected'].append(arr[0].copy())
        return raw_result

    model._build_broken_camera2lidar=shared_perturbation
    model._build_calib_dict_from_cam2lidar=capture_calib
    try:
        with torch.no_grad(): result=model(inp,samples,mode='predict')[0]
    finally:
        model._build_broken_camera2lidar=original_broken_builder
        model._build_calib_dict_from_cam2lidar=original_build
    boxes=get_boxes(result)
    if observed['called'] != 1:
        raise RuntimeError('Official LCCNet did not request shared perturbation exactly once')
    if len(observed['corrected']) < 2:
        raise RuntimeError('Could not intercept Official LCCNet corrected camera extrinsic')
    broken_off=observed['broken']
    if not np.allclose(broken_off,pcc_br,atol=4e-3):
        raise RuntimeError('Shared Broken extrinsic differs after Official inference')
    # The Official lccnet code calls build_calib_dict(broken), then
    # build_calib_dict(corrected). Pick the final call that is neither the
    # exact broken nor clean state; log all candidate distances.
    options=[]
    for i,cand in enumerate(observed['corrected']):
        db=float(np.max(np.abs(cand-broken_off)))
        dc=float(np.max(np.abs(cand-orig)))
        options.append((i,db,dc))
    valid=[i for i,db,_dc in options if db>1e-6]
    if not valid:
        raise RuntimeError('No LCCNet correction distinct from Broken was captured')
    corrected=observed['corrected'][valid[-1]]
    save_npz(dirpath/'official.npz',boxes,clean=orig,broken=broken_off,
             K=k_official,recovered=corrected)
    write_json(dirpath/'official_meta.json',dict(frame_key=ref['frame_key'],
        token=str(meta.get('token',meta.get('sample_idx',''))),img_paths=paths,
        lidar_path=data_sample_path(meta),dataset_index=int(idx),
        official_config=args.official_config,official_checkpoint=args.official_checkpoint,
        lccnet_model_checkpoint=str(model.lccnet_checkpoint),
        clean_matrix_max_diff=gt_diff, intrinsics_max_diff=k_diff,
        build_calib_candidates=options,observed_perturbation_calls=observed['called']))
    print('[OFFICIAL+LCCNet] matched_idx=',idx,'boxes=',len(boxes['scores']),
          'capture candidates=',options)


def save_npz(path,boxes,**matrices):
    data={k:np.asarray(v) for k,v in boxes.items()}
    data.update({k:np.asarray(v) for k,v in matrices.items()})
    np.savez_compressed(str(path),**data)


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')


def project_xyz(points,T_c2l,K,img_hw,max_depth=80):
    """Original LiDAR -> original camera pixels. Returns uv, front-facing mask."""
    H,W=img_hw
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if len(points)==0: return np.empty((0,2)),np.zeros(0,dtype=bool)
    P=np.concatenate((points,np.ones((len(points),1))),axis=1)
    Tc=np.asarray(T_c2l,dtype=np.float64).reshape(4,4)
    Pc=(np.linalg.inv(Tc) @ P.T).T[:,:3]
    z=Pc[:,2]
    K3=np.asarray(K,dtype=np.float64)[:3,:3]
    proj=(K3 @ Pc.T).T
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=proj[:,:2]/proj[:,2:3]
    mask=(np.isfinite(uv).all(axis=1) & (z>0.3) & (z<max_depth) &
          (uv[:,0]>=0) & (uv[:,0]<W) & (uv[:,1]>=0) & (uv[:,1]<H))
    return uv,mask


def raw_keyframe_points(path,args):
    path=resolve_path(path)
    if path.endswith('.npy'):
        pts=np.asarray(np.load(path))
    else:
        raw=np.fromfile(path,dtype=np.float32)
        if raw.size%5: raise RuntimeError('nuScenes LIDAR_TOP raw .bin not divisible by 5')
        pts=raw.reshape(-1,5)
    xyz=np.asarray(pts[:,:3],dtype=np.float64)
    d=np.linalg.norm(xyz[:,:2],axis=1)
    mask=(np.isfinite(xyz).all(axis=1)&(d>=args.min_range)&
          (d<=args.max_range)&(xyz[:,2]>-3.0)&(xyz[:,2]<6.0))
    return xyz[mask]


def _sample25_enhanced(args):
    """Enhance only validation index 25, never other sample indices."""
    return int(args.sample_idx) == 25 and not args.sample25_no_enhance


def _select_common_projection_ids(pts, gt, K, hw, args):
    """Select LiDAR point IDs ONCE in the GT view and reuse for all 3 poses.

    This ensures that any visible displacement reflects calibration only.
    """
    _, in_view = project_xyz(pts, gt, K, hw, args.max_depth)
    ids = np.flatnonzero(in_view)
    if len(ids) > args.max_points:
        rng = np.random.default_rng(args.seed + args.sample_idx)
        ids = np.sort(rng.choice(ids, size=args.max_points, replace=False))
    if _sample25_enhanced(args):
        step = max(1, int(args.sample25_stride))
        ids = ids[::step]
    return ids


def _sample25_paint_points(bgr, uv, color, args, *, gt=False, standalone=False):
    """Efficient sample-25 glow/outline rendering via ONE mask per point class.

    Only pixels and styles change. 3D inputs, point IDs, projections, model
    inference, detection boxes, and camera intrinsics are never modified.

    OpenCV colors are BGR. No geometric offset or synthetic point is added.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    H, W = bgr.shape[:2]
    if uv.shape[0] == 0:
        return bgr
    uv = uv[np.isfinite(uv).all(axis=1)]
    coords = np.rint(uv).astype(np.int32)
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < W) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < H)]
    if len(coords) == 0:
        return bgr

    core_r = max(1, int(args.sample25_core_radius) - (1 if gt else 0))
    border_r = max(core_r + 1, int(args.sample25_outline_radius))
    sigma = max(0.0, float(args.sample25_halo_sigma))
    halo_alpha = float(np.clip(args.sample25_glow_alpha, 0.0, 1.0))

    # One OpenCV mask per point class, not a whole-frame copy per LiDAR point.
    pin = np.zeros((H, W), dtype=np.uint8)
    for x, y in coords:
        cv2.circle(pin, (int(x), int(y)), 1, 255, -1, cv2.LINE_AA)

    out = bgr.astype(np.float32, copy=True)
    col = np.asarray(color, dtype=np.float32)

    if sigma > 0 and halo_alpha > 0:
        bloom = cv2.GaussianBlur(pin.astype(np.float32) / 255.0,
                                 (0, 0), sigmaX=sigma, sigmaY=sigma)
        # In dense regions cap bloom so that the underlying photo stays visible.
        halo = np.minimum(bloom * 2.5 * halo_alpha, 0.30)[..., None]
        out += (col[None, None, :] - out) * halo

    cores = np.zeros((H, W), dtype=np.uint8)
    for x, y in coords:
        cv2.circle(cores, (int(x), int(y)), core_r, 255, -1, cv2.LINE_AA)
    diameter = 2 * border_r + 1
    outline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (diameter, diameter))
    dilated = cv2.dilate(pin, outline_kernel)
    # Dark rim drawn before the colored cores. Preserve camera context.
    # Do not erase the yellow recovered ring with GT's black outline when
    # the recovered coordinates coincide with GT. The green GT core is kept.
    rim_strength = 0.48 if (not gt or standalone) else 0.04
    rim = (dilated.astype(np.float32) / 255.0 * rim_strength)[..., None]
    out *= (1.0 - rim)
    center = (cores.astype(np.float32) / 255.0 * 0.96)[..., None]
    out += (col[None, None, :] - out) * center
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def plot_overlay(raw, pts, gt, broken, recovered, K, args):
    """Draw GT, Broken, and Recovered from the SAME selected LiDAR point IDs."""
    H, W = raw.shape[:2]
    ids = _select_common_projection_ids(pts, gt, K, (H, W), args)
    xyz = pts[ids]
    visibility = {}
    if _sample25_enhanced(args):
        # Paint Broken -> Recovered -> GT. A small green GT center remains
        # visible when yellow recovered and green GT coincide.
        image = raw.copy()
        for label, T, color in (
            ('Broken', broken, BROKEN_COLOR),
            ('Recovered', recovered, RECOVER_COLOR),
            ('GT', gt, GT_COLOR),
        ):
            uv, mask = project_xyz(xyz, T, K, (H, W), args.max_depth)
            visibility[label] = int(mask.sum())
            image = _sample25_paint_points(image, uv[mask], color, args,
                                           gt=(label == 'GT'))
        return image, int(len(ids)), visibility

    image = raw.copy()
    canvas = image.copy()
    for label, T, color in (
        ('Broken', broken, BROKEN_COLOR),
        ('GT', gt, GT_COLOR),
        ('Recovered', recovered, RECOVER_COLOR),
    ):
        uv, mask = project_xyz(xyz, T, K, (H, W), args.max_depth)
        visibility[label] = int(mask.sum())
        for u, v in np.rint(uv[mask]).astype(np.int32):
            cv2.circle(canvas, (int(u), int(v)), args.point_radius, color,
                       -1, cv2.LINE_AA)
    out = cv2.addWeighted(canvas, 0.88, image, 0.12, 0)
    return out, int(len(ids)), visibility


def draw_bbox_image(raw,boxes,clean,K,args,box_color):
    H,W=raw.shape[:2]
    img=raw.copy()
    order=np.argsort(-boxes['scores'])
    count=0
    for j in order:
        score=float(boxes['scores'][j])
        if score<args.score_thr or count>=args.max_boxes: break
        cor=boxes['corners'][j]
        uv,front=project_xyz(cor,clean,K,(H,W),args.max_depth)
        if not bool(front.any()): continue
        # Entire box behind or crossing the near plane is skipped; never
        # draw long spurious lines from perspective division at z<=0.
        homog=np.concatenate((cor,np.ones((8,1))),axis=1)
        zz=(np.linalg.inv(clean) @ homog.T).T[:,2]
        if float(np.min(zz)) <= .30 or float(np.max(zz))>args.max_depth:
            continue
        # An edge is rendered only when both of its endpoints are in view.
        valid_edges=0
        for a,b in BOX_EDGES:
            if front[a] and front[b]:
                cv2.line(img,tuple(np.rint(uv[a]).astype(int)),
                         tuple(np.rint(uv[b]).astype(int)),box_color,2,cv2.LINE_AA)
                valid_edges+=1
        if valid_edges == 0: continue
        label=int(boxes['labels'][j]);name=CLASSES[label] if 0<=label<len(CLASSES) else str(label)
        valid_uv=uv[front]
        x=int(np.clip(np.min(valid_uv[:,0]),0,W-1))
        y=int(np.clip(np.min(valid_uv[:,1])-6,12,H-8))
        cv2.putText(img,f'{name} {score:.2f}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,
                    .43,box_color,1,cv2.LINE_AA)
        count+=1
    return img,count


def banner(im,title,subtitle='',legend=False,scale=1,legend_entries=None):
    h,w=im.shape[:2]
    top=76 if legend else 53
    out=np.full((h+top,w,3),(27,28,30),dtype=np.uint8)
    out[top:]=im
    cv2.putText(out,title,(14,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),2,cv2.LINE_AA)
    if legend:
        entries=legend_entries if legend_entries is not None else [
            ('GT',GT_COLOR),('Broken',BROKEN_COLOR),('Recovered',RECOVER_COLOR)]
        x=17
        for name,color in entries:
            cv2.circle(out,(x,47),5,color,-1,cv2.LINE_AA)
            cv2.putText(out,name,(x+12,51),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
            x+={'GT':120,'Broken':155,'Recovered':180}[name]
    if subtitle:
        yy=67 if legend else 44
        cv2.putText(out,subtitle[:95],(14,yy),cv2.FONT_HERSHEY_SIMPLEX,.38,(190,200,210),1,cv2.LINE_AA)
    if scale>1:
        out=cv2.resize(out,(out.shape[1]*scale,out.shape[0]*scale),interpolation=cv2.INTER_CUBIC)
    return out


# ================================================================
# nuScenes GT (separate from any model prediction)
# ================================================================

def detection_class_from_nuscenes(category):
    """Map raw nuScenes annotation categories to the 10 detection classes.

    Unmapped categories are ignored deliberately (the evaluation protocol also
    excludes these classes). The exact raw name remains in the annotation table.
    """
    name=str(category)
    if name.startswith('human.pedestrian.'):
        return 'pedestrian'
    if name.startswith('vehicle.bus.'):
        return 'bus'
    for prefix, output in (
        ('vehicle.car', 'car'),
        ('vehicle.truck', 'truck'),
        ('vehicle.construction', 'construction_vehicle'),
        ('vehicle.trailer', 'trailer'),
        ('vehicle.motorcycle', 'motorcycle'),
        ('vehicle.bicycle', 'bicycle'),
        ('movable_object.barrier', 'barrier'),
        ('movable_object.trafficcone', 'traffic_cone'),
    ):
        if name == prefix or name.startswith(prefix+'.'):
            return output
    return None


def load_matched_nuscenes_gt(pmeta, camera_name, image_path, K_expected,
                            nuscenes_root, nuscenes_version, image_hw):
    """Load GT 3D camera-frame boxes for the *exact* PCC keyframe.

    The official nuScenes API transforms global GT to the camera frame, taking
    camera/ego timestamps into account. We verify LIDAR_TOP filename,
    CAM image basename and intrinsics before allowing GT to be displayed.
    The returned corners are in original camera coordinates, NOT LiDAR coords.
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import BoxVisibility
    except ImportError as e:
        raise RuntimeError(
            'nuScenes devkit missing. Install nuscenes-devkit in the same '
            'environment and verify the v1.0-trainval annotation files.') from e

    root=Path(nuscenes_root).expanduser()
    version_dir=root/nuscenes_version
    if not version_dir.is_dir():
        raise FileNotFoundError(
            f'nuScenes GT tables not found at {version_dir}. '
            'Pass --nuscenes-root and --nuscenes-version; GT needs annotations.')

    nusc=NuScenes(version=nuscenes_version,dataroot=str(root),verbose=False)
    candidate=str(pmeta.get('token',''))
    target_frame=pmeta['frame_key']
    expected_cam_name=Path(image_path).name

    def is_exact_frame(s):
        sd=nusc.get('sample_data',s['data']['LIDAR_TOP'])
        return Path(sd['filename']).name == target_frame

    matched=None
    if len(candidate)==32:
        try:
            maybe=nusc.get('sample',candidate)
            if is_exact_frame(maybe):
                matched=maybe
        except (KeyError,ValueError,TypeError):
            pass

    if matched is None:
        # Do not trust the numeric dataset index; the two info files can differ.
        matches=[s for s in nusc.sample if is_exact_frame(s)]
        if len(matches)!=1:
            raise RuntimeError(
                f'Could not uniquely match GT for LiDAR file {target_frame}: '
                f'{len(matches)} sample(s) found')
        matched=matches[0]

    cam_sd_token=matched['data'][camera_name]
    cam_sd=nusc.get('sample_data',cam_sd_token)
    if Path(cam_sd['filename']).name!=expected_cam_name:
        raise RuntimeError(
            'GT camera sample differs from detector camera image: '
            f'{cam_sd["filename"]} vs {image_path}')

    _,sdk_boxes,K_api=nusc.get_sample_data(
        cam_sd_token,box_vis_level=BoxVisibility.ANY)
    K_api=np.asarray(K_api,dtype=np.float64)
    K_model=np.asarray(K_expected,dtype=np.float64)[:3,:3]
    max_K_diff=float(np.max(np.abs(K_api-K_model)))
    if max_K_diff>3e-3:
        raise RuntimeError(
            'GT SDK original camera K differs from model clean intrinsics: '
            f'max_abs_diff={max_K_diff:.6g}; check camera order/augmented K')

    H,W=image_hw
    if int(cam_sd.get('height',H))!=H or int(cam_sd.get('width',W))!=W:
        raise RuntimeError(
            'Camera image dimensions differ from nuScenes GT sample_data: '
            f'jpg={(W,H)} vs SDK={(cam_sd.get("width"),cam_sd.get("height"))}')

    items=[]
    counts_unmapped={}
    for gtbox in sdk_boxes:
        raw_name=getattr(gtbox,'name','')
        class_name=detection_class_from_nuscenes(raw_name)
        if class_name is None:
            counts_unmapped[raw_name]=counts_unmapped.get(raw_name,0)+1
            continue
        corners=np.asarray(gtbox.corners(),dtype=np.float64).T  # [8,3] camera frame
        if corners.shape!=(8,3) or not np.isfinite(corners).all():
            raise RuntimeError(f'Invalid GT box corners for {raw_name}')
        items.append({'class':class_name,'raw_category':raw_name,
                      'corners_cam':corners,'depth':float(gtbox.center[2])})
    items.sort(key=lambda x:x['depth'],reverse=True) # far first; near objects on top
    return items, {
        'nuscenes_sample_token':matched['token'],
        'camera_sample_data_token':cam_sd_token,
        'matched_lidar_filename':target_frame,
        'matched_camera_filename':expected_cam_name,
        'intrinsic_max_abs_diff':max_K_diff,
        'annotated_3d_boxes_in_camera':len(sdk_boxes),
        'gt_detection_class_boxes_in_camera':len(items),
        'ignored_categories':counts_unmapped,
        'gt_annotation_source':'nuScenes SDK sample_annotation in camera coordinates',
    }


def draw_gt_bbox_image(raw, gt_items, K, max_depth, color=GT_BOX_COLOR):
    """Draw raw nuScenes GT boxes, with no prediction score threshold.

    3D edges are rendered only if both endpoints have positive depth and lie
    inside the image. Unlike an unchecked perspective projection, this will
    not connect corners behind the camera across the whole image.
    """
    img=raw.copy()
    H,W=img.shape[:2]
    k=np.asarray(K,dtype=np.float64)[:3,:3]
    drawn=0
    for item in gt_items:
        xyz=item['corners_cam']
        z=xyz[:,2]
        if float(np.min(z))<=0.3 or float(np.max(z))>max_depth:
            continue
        pixels=(k@xyz.T).T
        uv=pixels[:,:2]/pixels[:,2:3]
        visible=(np.isfinite(uv).all(axis=1) & (z>0.3) &
                 (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H))
        drawn_edges=0
        for a,b in BOX_EDGES:
            if visible[a] and visible[b]:
                p0=tuple(np.rint(uv[a]).astype(int))
                p1=tuple(np.rint(uv[b]).astype(int))
                cv2.line(img,p0,p1,color,2,cv2.LINE_AA)
                drawn_edges+=1
        if not drawn_edges:
            continue
        xy=uv[visible]
        x=int(np.clip(np.min(xy[:,0]),0,W-1))
        y=int(np.clip(np.min(xy[:,1])-6,12,H-8))
        cv2.putText(img,item['class'],(x,y),cv2.FONT_HERSHEY_SIMPLEX,
                    .45,color,1,cv2.LINE_AA)
        drawn+=1
    return img,drawn


def crop_all_equally(images, roi):
    if roi is None: return [im.copy() for im in images]
    x0,y0,x1,y1=roi
    H,W=images[0].shape[:2]
    if any(im.shape[:2]!=(H,W) for im in images):
        raise RuntimeError('Images must share original dimensions before paper crop')
    if not (0<=x0<x1<=W and 0<=y0<y1<=H):
        raise ValueError(f'Invalid --paper-crop {roi} for original size {W}x{H}')
    return [im[y0:y1,x0:x1].copy() for im in images]


def single_bev(points,boxes,color,span=55.0,side=660):
    canvas=np.zeros((side,side,3),np.uint8)
    def cvxy(x,y):
        u=(.5-y/(2*span))*side
        v=(1.0-(x+span)/(2*span))*side
        return int(round(u)),int(round(v))
    xy=points[:,:2]
    pick=(np.abs(xy[:,0])<=span)&(np.abs(xy[:,1])<=span)
    sub=xy[pick][::4]
    for x,y in sub:
        u,v=cvxy(x,y)
        if 0<=u<side and 0<=v<side: canvas[v,u]=(80,80,80)
    cv2.line(canvas,cvxy(-span,0),cvxy(span,0),(50,50,50),1)
    cv2.line(canvas,cvxy(0,-span),cvxy(0,span),(50,50,50),1)
    for j in np.argsort(-boxes['scores'])[:50]:
        if boxes['scores'][j]<.25: continue
        cor=boxes['corners'][j]
        poly=[cvxy(*cor[k,:2]) for k in (0,2,6,4)]
        cv2.polylines(canvas,[np.asarray(poly,dtype=np.int32)],True,color,2,cv2.LINE_AA)
    return canvas


def compose_worker_outputs(args,dirpath):
    pmeta=json.loads((dirpath/'pcc_meta.json').read_text())
    ometa=json.loads((dirpath/'official_meta.json').read_text())
    p=np.load(str(dirpath/'pcc.npz'))
    o=np.load(str(dirpath/'official.npz'))
    if pmeta['frame_key']!=ometa['frame_key']:
        raise RuntimeError('Different physical frame in worker results')
    if max(np.max(np.abs(p['clean']-o['clean'])),
           np.max(np.abs(p['broken']-o['broken'])))>4e-3:
        raise RuntimeError('Official/PCC images have different underlying calibration')
    idx=CAMERAS.index(args.cam_name)
    image_path=resolve_path(pmeta['img_paths'][idx])
    original=cv2.imread(image_path,cv2.IMREAD_COLOR)
    if original is None: raise RuntimeError(f'Cannot open {image_path}')
    original=np.ascontiguousarray(original)
    gt=p['clean'][idx];broken=p['broken'][idx];K=p['K'][idx]
    # Test against dataset calibration matrix whenever available.
    if max(np.abs(p['K'][idx]-o['K'][idx]).flatten()) > 3e-3:
        raise RuntimeError('Image intrinsics differ across model runs')
    lidar_points=raw_keyframe_points(pmeta['lidar_path'],args)
    gt_uv,gt_mask=project_xyz(lidar_points,gt,K,original.shape[:2],args.max_depth)
    print('[GT gate] visible keyframe points before subsampling =',int(gt_mask.sum()))
    if gt_mask.sum()<80: raise RuntimeError('GT projection unexpectedly sparse: inspect extrinsic and LiDAR path')
    stem=f'sample{args.sample_idx:05d}_{args.cam_name}'
    gt_only = original.copy()
    if _sample25_enhanced(args):
        # Reuse exactly the same GT-visible point IDs as plot_overlay().
        selected_ids = _select_common_projection_ids(
            lidar_points, gt, K, original.shape[:2], args)
        gt_only = _sample25_paint_points(
            gt_only, gt_uv[selected_ids], GT_COLOR, args, gt=True, standalone=True)
        print('[SAMPLE25] Enhanced GT / Broken / Recovered projection points: '
              f'{len(selected_ids)} matched GT-visible LiDAR returns; '
              f'stride={args.sample25_stride}, '
              f'core_radius={args.sample25_core_radius}, '
              f'outline_radius={args.sample25_outline_radius}', flush=True)
    else:
        for u, v in np.rint(gt_uv[gt_mask][::max(1, int(gt_mask.sum())//3000)]).astype(np.int32):
            cv2.circle(gt_only, (int(u), int(v)), 1, GT_COLOR, -1)
    cv2.imwrite(str(dirpath/f'{stem}_GT_only.png'),gt_only)
    candidates=(('lccnet_official',o,'LCCNet + Official BEVFusion',OFFICIAL_BOX_COLOR),
                ('pcc',p,'Ours: PCC',PCC_BOX_COLOR))
    panels=[]; detection_panels=[]; stats={}
    for name,blob,label,bcol in candidates:
        lcal=blob['recovered'][idx]
        im_top,n_points,visible=plot_overlay(original,lidar_points,gt,broken,lcal,K,args)
        predicted={k:blob[k] for k in ('corners','boxes','scores','labels')}
        im_bottom,n_boxes=draw_bbox_image(original,predicted,gt,K,args,bcol)
        rowtop=banner(im_top,label+' | calibration projection',
                      f'Same LiDAR points, same perturbation | GT in-view: {n_points}',
                      legend=True,scale=args.scale)
        rowbot=banner(im_bottom,label+' | 3D detections',
                      f'Projected with CLEAN camera extrinsics | score >= {args.score_thr:.2f}',
                      legend=False,scale=args.scale)
        stacked=np.vstack((rowtop,rowbot))
        saveimg(dirpath / (f'{stem}_' + ('01_lccnet_official' if name=='lccnet_official' else '02_pcc') + '.png'), stacked)
        panels.append(stacked)
        detection_panels.append(rowbot)
        stats[name]={'boxes_above_score_threshold':int(np.sum(blob['scores']>=args.score_thr)),
                     'projected_boxes_drawn':n_boxes,'gt_selected_points':n_points,
                     'point_visibility':visible}
    merged=np.hstack(panels)
    saveimg(dirpath/f'{stem}_comparison_2col.png',merged)
    detonly=np.hstack(detection_panels)
    saveimg(dirpath/f'{stem}_detection_only_2col.png',detonly)
    # ---------------------------------------------------------------
    # NEW: ground-truth 3D boxes from official nuScenes annotations
    # ---------------------------------------------------------------
    gt_items,gt_meta=load_matched_nuscenes_gt(
        pmeta,args.cam_name,image_path,K,
        args.nuscenes_root,args.nuscenes_version,original.shape[:2])
    im_gt,gt_drawn=draw_gt_bbox_image(original,gt_items,K,args.max_depth)
    if gt_drawn==0:
        raise RuntimeError('No visible 3D GT boxes in this frame/camera. '
                           'Select a different sample for a meaningful paper comparison.')
    print(f'[GT] nuScenes token={gt_meta["nuscenes_sample_token"]} '
          f'boxes_in_view={len(gt_items)} rendered={gt_drawn}')
    saveimg(dirpath/f'{stem}_GT_3D_boxes.png', im_gt)
    # The same single-frame LiDAR set is shown in the GT overlay, without a
    # third "recovered" transform. This row is diagnostic; detection-only 3col
    # below is the more compact figure suitable for an 8-page manuscript.
    gt_bottom=banner(im_gt,'Ground truth | 3D annotated boxes',
                     f'nuScenes 10 detection classes | drawn: {gt_drawn}',
                     legend=False,scale=args.scale)
    # Equal header height with prediction panels; GT does not have a Broken
    # or Recovered series. This custom legend prevents misleading labeling.
    gt_top=banner(gt_only,'GT | clean LiDAR projection',
                  'Original camera image, clean extrinsics',
                  legend=True,scale=args.scale,
                  legend_entries=[('GT',GT_COLOR)])
    gt_full=np.vstack((gt_top,gt_bottom))
    if gt_full.shape[:2] != panels[0].shape[:2]:
        raise RuntimeError(f'GT full panel {gt_full.shape} differs from predictions {panels[0].shape}')

    gt_det_panel = banner(im_gt,'(a) GT 3D boxes',
                          'nuScenes annotation (10 detection classes)',
                          legend=False,scale=args.scale)
    if gt_det_panel.shape[:2] != detection_panels[0].shape[:2]:
        raise RuntimeError('GT detection panel dimensions do not match predictions')

    # By default, publish original-resolution images. A user-requested ROI is
    # applied consistently to *all three* lower-row camera photos; no detections
    # are selectively redrawn after cropping.
    if args.paper_crop is None:
        paper_gt,paper_lcc,paper_pcc=gt_det_panel,detection_panels[0],detection_panels[1]
        full_gt,full_lcc,full_pcc=gt_full,panels[0],panels[1]
    else:
        crop=args.paper_crop
        raw_gt,raw_lcc,raw_pcc=crop_all_equally(
            [im_gt,
             draw_bbox_image(original,{k:o[k] for k in ('corners','boxes','scores','labels')},gt,K,args,OFFICIAL_BOX_COLOR)[0],
             draw_bbox_image(original,{k:p[k] for k in ('corners','boxes','scores','labels')},gt,K,args,PCC_BOX_COLOR)[0]],
            crop)
        paper_gt=banner(raw_gt,'(a) GT 3D boxes',
                        'nuScenes ground truth',scale=args.scale)
        paper_lcc=banner(raw_lcc,'(b) LCCNet + Official BEVFusion',
                         f'score >= {args.score_thr:.2f}',scale=args.scale)
        paper_pcc=banner(raw_pcc,'(c) PCC (Ours)',
                         f'score >= {args.score_thr:.2f}',scale=args.scale)
        gt_calib=crop_all_equally([gt_only],crop)[0]
        calib_lcc=crop_all_equally([plot_overlay(original,lidar_points,gt,broken,o['recovered'][idx],K,args)[0]],crop)[0]
        calib_pcc=crop_all_equally([plot_overlay(original,lidar_points,gt,broken,p['recovered'][idx],K,args)[0]],crop)[0]
        full_gt=np.vstack((banner(gt_calib,'GT | clean LiDAR',legend=True,scale=args.scale,
                                      legend_entries=[('GT',GT_COLOR)]),paper_gt))
        full_lcc=np.vstack((banner(calib_lcc,'LCCNet | calibration',legend=True,scale=args.scale),paper_lcc))
        full_pcc=np.vstack((banner(calib_pcc,'PCC | LGPC calibration',legend=True,scale=args.scale),paper_pcc))

    three_det=np.hstack((paper_gt,paper_lcc,paper_pcc))
    saveimg(dirpath/f'{stem}_detection_only_3col_GT_LCCNet_PCC.png',three_det)
    three_full=np.hstack((full_gt,full_lcc,full_pcc))
    saveimg(dirpath/f'{stem}_comparison_3col_GT_LCCNet_PCC.png',three_full)
    stats['ground_truth']={'gt_detection_class_boxes_in_camera':len(gt_items),
                           'gt_boxes_drawn':gt_drawn}
    if args.with_bev:
        bev_panels=[]
        for name,blob,label,bcol in candidates:
            box={k:blob[k] for k in ('corners','boxes','scores','labels')}
            im=single_bev(lidar_points,box,bcol)
            # Match per-column widths of the stacked camera panels.
            w=panels[0].shape[1]
            im=cv2.resize(im,(w,w),interpolation=cv2.INTER_NEAREST)
            bev_panels.append(banner(im,label+' | BEV (predictions only)',scale=1))
        bev=np.hstack(bev_panels)
        saveimg(dirpath/f'{stem}_comparison_2col_bev.png',np.vstack((merged,bev)))
    write_json(dirpath/f'{stem}_manifest.json',{
        'frame_key':pmeta['frame_key'],'camera':args.cam_name,
        'pcc_val_idx':args.sample_idx,'official_val_idx':ometa['dataset_index'],
        'source_photo':image_path,
        'same_photo_verified':True,'same_camera_gt_and_broken_verified':True,
        'shared_delta_convention':'Delta = PCC_broken @ inverse(PCC_clean), Official_broken = Delta @ Official_clean',
        'projection_reference':'All 3D detection boxes projected with CLEAN camera extrinsic',
        'PCC_calibration_overlay':'LGPC Stage-1 predicted correction',
        'Official_calibration_overlay':'Actual LCCNet prediction intercepted during Official forward',
        'RRRF_note':'RRRF does not directly reproject whole LiDAR cloud; only its downstream 3D detection boxes shown',
        'points':'RAW single LiDAR keyframe; GT-visible points selected before limiting maximum',
        'qualitative_only':'No TP/FP/FN, mAP, or NDS is derived from these panels',
        'metric_warning':'Figure forces matched PCC perturbation into Official inference. It is NOT the original unpaired Official/LCCNet metric run.',
        'ground_truth':gt_meta,
        'three_column_layout':['GT 3D boxes','LCCNet + Official BEVFusion','PCC (Ours)'],
        'three_column_crop':args.paper_crop,
        'ground_truth_note':'GT boxes are raw nuScenes annotations, not model predictions; only 10 detection classes shown.',
        'performance':stats,
        'pcc':pmeta,'official':ometa})
    print('[DONE] Paper comparison panels saved under',dirpath)


def saveimg(path,img):
    if not cv2.imwrite(str(path),img):
        raise RuntimeError(f'Failed to save {path}')
    print('[SAVE]',path)


def verify_reusable_worker_cache(args,dirpath):
    """Fail rather than silently reusing a different frame/checkpoint.

    For the exact same old two-worker run, both .npz files can be reused
    to add a GT column without another expensive BEVFusion inference.
    """
    needed=('pcc.npz','official.npz','pcc_meta.json','official_meta.json')
    for name in needed:
        verify_exists(dirpath/name,'Worker cache')
    pmeta=json.loads((dirpath/'pcc_meta.json').read_text())
    ometa=json.loads((dirpath/'official_meta.json').read_text())
    pairs=(('PCC sample index',pmeta.get('dataset_index'),args.sample_idx),
           ('PCC config',pmeta.get('pcc_config'),args.pcc_config),
           ('PCC weights',pmeta.get('pcc_checkpoint'),args.pcc_checkpoint),
           ('Official config',ometa.get('official_config'),args.official_config),
           ('Official weights',ometa.get('official_checkpoint'),args.official_checkpoint))
    if args.official_sample_idx>=0:
        pairs+= (('Official sample index',ometa.get('dataset_index'),args.official_sample_idx),)
    if args.lccnet_model_checkpoint:
        pairs+= (('LCCNet weights',ometa.get('lccnet_model_checkpoint'),args.lccnet_model_checkpoint),)
    for name,old,new in pairs:
        if str(old)!=str(new):
            raise RuntimeError(f'Worker cache mismatch for {name}: saved={old!r}, requested={new!r}; '
                               'remove --reuse-worker-cache and rerun actual inference')
    if pmeta.get('frame_key')!=ometa.get('frame_key'):
        raise RuntimeError('Worker cache contains different LiDAR frames')
    print('[CACHE] PCC and Official predictions safely reused for same frame/checkpoints.')


def worker_command(args,mode):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--run-worker',mode]
    # Pass identical effective args to both workers and final composer.
    for name in ('pcc-config','pcc-checkpoint','official-config','official-checkpoint',
                 'lccnet-model-checkpoint','cam-name','out-dir','device'):
        value=getattr(args,name.replace('-','_'))
        if value!='': cmd.extend(['--'+name,str(value)])
    for name in ('sample-idx','official-sample-idx','seed'):
        cmd.extend(['--'+name,str(getattr(args,name.replace('-','_')))])
    return cmd


# ================================================================
# AUTOMATIC QUALITATIVE FRAME RANKING (ground-truth based)
# ================================================================

def _path_fingerprint(path):
    if not path:
        return None
    obj=Path(path)
    if not obj.is_file():
        return {'path': str(path), 'exists': False}
    stat=obj.stat()
    return {'path': str(obj.resolve()), 'bytes': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}


def _cache_signature(args):
    """Verify model files and runtime configs, not only their filenames."""
    return {
        'pcc_config': _path_fingerprint(args.pcc_config),
        'pcc_checkpoint': _path_fingerprint(args.pcc_checkpoint),
        'official_config': _path_fingerprint(args.official_config),
        'official_checkpoint': _path_fingerprint(args.official_checkpoint),
        'lccnet_model_checkpoint': _path_fingerprint(args.lccnet_model_checkpoint),
        'paired_worker_script': _path_fingerprint(__file__),
        'seed': args.seed,
        'device': args.device,
    }


def _is_figure_renderable_box(corners,clean,K,image_hw,max_depth):
    """Use the same visibility checks as the 3D box visualization."""
    xyz=np.asarray(corners,dtype=np.float64)
    if xyz.shape != (8,3) or not np.isfinite(xyz).all():
        return False
    ch=np.c_[xyz,np.ones(8)]
    depths=(np.linalg.inv(clean) @ ch.T).T[:,2]
    if float(np.min(depths))<=0.30 or float(np.max(depths))>max_depth:
        return False
    _,inside=project_xyz(xyz,clean,K,image_hw,max_depth)
    return any(bool(inside[a] and inside[b]) for a,b in BOX_EDGES)


def _safe_match_predictions(pred_cache, cam_idx, clean, K, shape, args):
    """Return detection centers/classes of boxes whose projected geometry is visible.

    This defines a consistent *camera-visible subset*, not an official nuScenes
    evaluation filter. Both detectors are treated the same way.
    """
    H,W=shape
    scores=pred_cache['scores']
    labels=pred_cache['labels']
    corners=pred_cache['corners']
    active=np.argsort(-scores)
    out=[]
    for j in active:
        if float(scores[j])<args.score_thr:
            break
        category_idx=int(labels[j])
        if not 0<=category_idx<len(CLASSES):
            continue
        xyz=np.asarray(corners[j],dtype=np.float64)
        if xyz.shape!=(8,3) or not np.isfinite(xyz).all():
            continue
        center=xyz.mean(axis=0)
        if not _is_figure_renderable_box(xyz,clean,K,(H,W),args.max_depth):
            continue
        out.append({'pred_idx':int(j),'class':CLASSES[category_idx],
                    'center_lidar':center,'confidence':float(scores[j])})
        if len(out)>=args.max_boxes:
            break
    return sorted(out,key=lambda o:-o['confidence'])


def _match_classwise_2m(gt_objects,pred_objects,threshold_m):
    """Confidence-prioritized greedy same-class 2D XY center-distance matches.

    These are diagnostics for selecting readable qualitative frames. They are
    explicitly NOT official nuScenes mAP, mATE or TP/FP/FN counts.
    """
    used=set()
    matches={}
    unmatched_pred=[]
    for p in pred_objects:
        c=np.asarray(p['center_lidar'],dtype=np.float64)
        options=[]
        for i,g in enumerate(gt_objects):
            if i in used or g['class'] != p['class']:
                continue
            gc=np.asarray(g['center_lidar'],dtype=np.float64)
            d=float(np.linalg.norm((gc-c)[:2]))
            if np.isfinite(d) and d<=threshold_m:
                options.append((d,i))
        if not options:
            unmatched_pred.append(p['pred_idx'])
            continue
        dist,gtidx=min(options)
        used.add(gtidx)
        matches[gtidx]={'prediction_index':p['pred_idx'],
                        'center_distance_m':dist,'class':p['class']}
    return dict(matches=matches,tp_proxy=len(matches),
                fp_proxy=len(unmatched_pred),
                fn_proxy=len(gt_objects)-len(matches),
                unmatched_predictions=unmatched_pred)


def _get_nuscenes_indexes(nusc):
    result={}
    for sample in nusc.sample:
        lidar_rec=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
        filename=Path(lidar_rec['filename']).name
        if filename in result:
            raise RuntimeError(f'Nonunique nuScenes LiDAR basename: {filename}')
        result[filename]=sample
    return result


def _load_gt_lidar_for_score(nusc,index_by_lidar, pcc_meta, pcc_blob, cam_name,
                             args):
    """Use annotations in the *LiDAR* coordinate system, same as pred boxes.

    This avoids measuring GT center in one sensor frame and prediction center
    in another. The nuScenes SDK handles global->ego->LiDAR transformation.
    """
    name=pcc_meta['frame_key']
    sample=index_by_lidar.get(name)
    if sample is None:
        raise RuntimeError(f'No nuScenes annotated sample for {name}')
    cam_idx=CAMERAS.index(cam_name)
    img_rec=nusc.get('sample_data',sample['data'][cam_name])
    img_expected=Path(pcc_meta['img_paths'][cam_idx]).name
    if Path(img_rec['filename']).name != img_expected:
        raise RuntimeError(f'Camera frame mismatch: {img_expected} vs {img_rec["filename"]}')
    cam_calib=nusc.get('calibrated_sensor',img_rec['calibrated_sensor_token'])
    sdk_K=np.asarray(cam_calib['camera_intrinsic'],dtype=np.float64)
    model_K=np.asarray(pcc_blob['K'][cam_idx],dtype=np.float64)[:3,:3]
    max_k_error=float(np.max(np.abs(sdk_K-model_K)))
    if max_k_error>3e-3:
        raise RuntimeError(f'GT/Model camera intrinsics mismatch: {max_k_error}')
    lidar_sd=sample['data']['LIDAR_TOP']
    _path,lidar_boxes,_intr=nusc.get_sample_data(lidar_sd)
    gt=[]
    clean=pcc_blob['clean'][cam_idx]
    image_shape=(int(img_rec['height']),int(img_rec['width']))
    for box in lidar_boxes:
        cls=detection_class_from_nuscenes(getattr(box,'name',''))
        if cls is None:
            continue
        corners=np.asarray(box.corners(),dtype=np.float64).T
        if corners.shape != (8,3) or not np.isfinite(corners).all():
            continue
        center=np.asarray(box.center,dtype=np.float64)
        if not _is_figure_renderable_box(corners,clean,model_K,image_shape,args.max_depth):
            continue
        gt.append({'class':cls,'center_lidar':center,
                   'annotation_token':str(getattr(box,'token',''))})
    return gt,sample['token'],sample['scene_token'],image_shape


def _score_count_focused(gt, lcc, ours, gain, lost, args):
    """Return an auditable paper-figure selection decision.

    Counts and match quantities refer only to the same original camera view.
    A count match alone is insufficient; one-to-one matched GTs must improve.
    This is purposefully an example selector, NOT an unbiased evaluation metric.
    """
    G=len(gt)
    P_lcc=int(lcc['tp_proxy']+lcc['fp_proxy'])
    P_ours=int(ours['tp_proxy']+ours['fp_proxy'])
    gap_lcc=abs(P_lcc-G)
    gap_ours=abs(P_ours-G)
    tp_gain=int(ours['tp_proxy']-lcc['tp_proxy'])
    excess_fp=int(ours['fp_proxy']-lcc['fp_proxy'])
    count_advantage=int(gap_lcc-gap_ours)
    # Difference in GT matches is counted only ONCE. FN improvement equals
    # TP improvement for fixed GT; do not award the same fact twice.
    score=(3.0*tp_gain + 2.0*count_advantage -
           1.5*max(excess_fp,0))

    reasons=[]
    if G < args.auto_min_visible_gt:
        reasons.append('too_few_visible_gt')
    if gap_ours > args.auto_max_ours_count_gap:
        reasons.append('pcc_count_not_close_to_gt')
    if gap_lcc < args.auto_min_lcc_count_gap:
        reasons.append('lcc_count_not_different_from_gt')
    if args.auto_lcc_under_detect and P_lcc > G-args.auto_min_lcc_count_gap:
        reasons.append('lcc_not_under_detecting')
    if tp_gain < args.auto_min_net_tp_gain:
        reasons.append('no_sufficient_pcc_net_gt_matches')
    if len(gain) < args.auto_min_gain:
        reasons.append('no_sufficient_pcc_unique_gt_recoveries')
    if ours['fp_proxy'] > args.auto_max_ours_fp:
        reasons.append('too_many_unmatched_pcc_predictions')
    if excess_fp > args.auto_max_extra_fp:
        reasons.append('too_many_extra_unmatched_pcc_predictions')
    if score<=0:
        reasons.append('score_not_positive')

    return dict(
        num_gt_visible=G,
        num_pred_lcc=P_lcc,
        num_pred_ours=P_ours,
        count_gap_lcc=gap_lcc,
        count_gap_ours=gap_ours,
        count_gap_advantage=count_advantage,
        net_tp_gain=tp_gain,
        pcc_extra_unmatched_predictions=excess_fp,
        selection_score=round(float(score),5),
        qualifies=not reasons,
        filter_reasons=';'.join(reasons) if reasons else 'PASS',
    )


def _rank_cached_frame(args, folder, nusc, index_by_lidar):
    verify_reusable_worker_cache(args,folder)
    pcc_meta=json.loads((folder/'pcc_meta.json').read_text())
    official_meta=json.loads((folder/'official_meta.json').read_text())
    with np.load(str(folder/'pcc.npz')) as p, np.load(str(folder/'official.npz')) as o:
        if max(float(np.max(np.abs(p['clean']-o['clean']))),
               float(np.max(np.abs(p['broken']-o['broken']))))>4e-3:
            raise RuntimeError('Unpaired calibration matrices in candidate frame')
        cam_idx=CAMERAS.index(args.cam_name)
        clean=np.asarray(p['clean'][cam_idx],dtype=np.float64)
        K=np.asarray(p['K'][cam_idx],dtype=np.float64)
        gt,token,scene,image_shape=_load_gt_lidar_for_score(
            nusc,index_by_lidar,pcc_meta,p,args.cam_name,args)
        lccpred=_safe_match_predictions(o,cam_idx,clean,K,image_shape,args)
        pccpred=_safe_match_predictions(p,cam_idx,clean,K,image_shape,args)
        lcc=_match_classwise_2m(gt,lccpred,args.auto_match_distance)
        ours=_match_classwise_2m(gt,pccpred,args.auto_match_distance)

    gain=set(ours['matches'])-set(lcc['matches'])
    lost=set(lcc['matches'])-set(ours['matches'])
    common=set(lcc['matches'])&set(ours['matches'])
    dist_gain=np.mean([lcc['matches'][i]['center_distance_m']-
                       ours['matches'][i]['center_distance_m'] for i in common]) if common else 0.0
    n_ped=sum(gt[i]['class']=='pedestrian' for i in gain)
    n_cone=sum(gt[i]['class']=='traffic_cone' for i in gain)
    decision=_score_count_focused(gt,lcc,ours,gain,lost,args)
    if decision['num_pred_lcc']!=len(lccpred) or decision['num_pred_ours']!=len(pccpred):
        raise RuntimeError('Visible prediction counts disagree with TP+FP proxy; check matching')

    return {
        'sample_idx':args.sample_idx,'frame_key':pcc_meta['frame_key'],
        'sample_token':token,'scene_token':scene,'camera':args.cam_name,
        'gt_visible':len(gt),
        'lcc_pred_visible':len(lccpred),'pcc_pred_visible':len(pccpred),
        'lcc_tp_proxy':lcc['tp_proxy'],'lcc_fp_proxy':lcc['fp_proxy'],
        'lcc_fn_proxy':lcc['fn_proxy'],'pcc_tp_proxy':ours['tp_proxy'],
        'pcc_fp_proxy':ours['fp_proxy'],'pcc_fn_proxy':ours['fn_proxy'],
        'tp_lcc':lcc['tp_proxy'],'tp_ours':ours['tp_proxy'],
        'fp_lcc':lcc['fp_proxy'],'fp_ours':ours['fp_proxy'],
        'fn_lcc':lcc['fn_proxy'],'fn_ours':ours['fn_proxy'],
        'gt_objects_recovered_only_by_pcc':len(gain),
        'gt_objects_recovered_only_by_lcc':len(lost),
        'pedestrian_gt_gain':n_ped,'traffic_cone_gt_gain':n_cone,
        'shared_gt_center_distance_improvement_m':round(float(dist_gain),5),
        **decision,
        'gt_gain_details':'; '.join(f'{gt[i]["class"]}:{gt[i]["annotation_token"]}' for i in sorted(gain)),
        'gt_loss_details':'; '.join(f'{gt[i]["class"]}:{gt[i]["annotation_token"]}' for i in sorted(lost)),
        'official_dataset_idx':official_meta['dataset_index'],
        'match_distance_m':args.auto_match_distance,
        'matched_perturbation_verified':True,
        'qualitative_proxy_not_official_nuscenes_metric':True,
    }


def _write_ranking_csv(path, rows):
    if not rows:
        return
    fields=list(rows[0].keys())
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _matching_inference_signature(saved, expected):
    """Reuse earlier worker predictions after rank-only changes to this script.

    Keep all checkpoint/config/seed/device fingerprints identical. An older
    script timestamp/filename may differ after code updates that change ONLY
    scoring/formatting. Per-frame metainfo+matrix validation is still required.
    """
    if not isinstance(saved,dict):
        return False
    return all(saved.get(key)==val for key,val in expected.items()
               if key!='paired_worker_script')


def _auto_sample_scan(args):
    import shutil
    if args.auto_rank_only and args.auto_force_rerun:
        raise ValueError('Cannot combine --auto-rank-only with --auto-force-rerun')
    if args.auto_search_count<1 or args.auto_search_step<1 or args.auto_top_k<1:
        raise ValueError('Positive --auto-search-count, --auto-search-step, --auto-top-k required')
    if args.auto_match_distance<=0:
        raise ValueError('--auto-match-distance must be positive')
    for name in ('auto_max_ours_count_gap','auto_min_lcc_count_gap',
                 'auto_min_net_tp_gain','auto_max_ours_fp','auto_max_extra_fp',
                 'auto_min_visible_gt','auto_min_gain'):
        if getattr(args,name)<0:
            raise ValueError(f'--{name.replace("_","-")} must be >= 0')
    if not 0<args.score_thr<1:
        raise ValueError('score threshold must be in (0,1)')
    verify_exists(args.pcc_checkpoint,'PCC checkpoint')
    verify_exists(args.official_checkpoint,'Official detector checkpoint')
    if args.lccnet_model_checkpoint:
        verify_exists(args.lccnet_model_checkpoint,'LCCNet model checkpoint')
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise RuntimeError('Install nuscenes-devkit to use GT-based auto selection') from exc
    nusc=NuScenes(version=args.nuscenes_version,dataroot=args.nuscenes_root,verbose=False)
    by_lidar=_get_nuscenes_indexes(nusc)
    root=Path(args.out_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    cache_root=root/'per_frame_cache'
    cache_root.mkdir(exist_ok=True)
    selection_root=root/'selected_top_k_count_focused'
    selection_root.mkdir(exist_ok=True)
    expected_sig=_cache_signature(args)
    results=[]
    failures=[]
    indices=[args.auto_search_start+i*args.auto_search_step
             for i in range(args.auto_search_count)]
    print(f'[AUTO] scan {len(indices)} PCC frames; GT matching within '
          f'{args.auto_match_distance:.2f}m, score>={args.score_thr:.2f}',flush=True)
    print('[AUTO] Count-focused: GT>=',args.auto_min_visible_gt,
          'PCC count gap<=',args.auto_max_ours_count_gap,
          'LCC count gap>=',args.auto_min_lcc_count_gap,
          'PCC net match gain>=',args.auto_min_net_tp_gain,
          'PCC FP<=',args.auto_max_ours_fp,
          'extra PCC FP<=',args.auto_max_extra_fp,flush=True)
    if args.auto_rank_only:
        print('[AUTO] --auto-rank-only: NO model inference, only verified caches.',flush=True)
    print('[AUTO] Favorable sample selection is qualitative ONLY, not official mAP/NDS!',flush=True)
    for ordinal,idx in enumerate(indices,1):
        one=copy.copy(args)
        one.sample_idx=int(idx)
        folder=cache_root/f'sample{idx:05d}'
        folder.mkdir(exist_ok=True)
        one.out_dir=str(folder)
        stamp=folder/'cache_signature.json'
        try:
            have_signature=stamp.is_file()
            saved_sig=json.loads(stamp.read_text()) if have_signature else None
            can_reuse=((args.auto_reuse_cache or args.auto_rank_only) and
                       not args.auto_force_rerun and have_signature and
                       _matching_inference_signature(saved_sig,expected_sig))
            if can_reuse:
                verify_reusable_worker_cache(one,folder)
                print(f'[AUTO {ordinal}/{len(indices)}] sample {idx}: verified cache',flush=True)
            elif args.auto_rank_only:
                # No silent reuse of stale/unidentified prediction artifacts.
                # Report missing frames and continue to the next cached frame.
                print(f'[AUTO {ordinal}/{len(indices)}] sample {idx}: '
                      'NO VERIFIED CACHE, skipping (inference disabled)',flush=True)
                failures.append({'sample_idx':idx,'type':'MissingOrChangedCache',
                                 'error':'No valid cache_signature + worker outputs with identical model/config/seed/device'})
                write_json(root/'errors.json',failures)
                continue
            else:
                print(f'[AUTO {ordinal}/{len(indices)}] sample {idx}: run PCC',flush=True)
                subprocess.run(worker_command(one,'pcc'),check=True)
                print(f'[AUTO {ordinal}/{len(indices)}] sample {idx}: run Official+LCCNet',flush=True)
                subprocess.run(worker_command(one,'official'),check=True)
                verify_reusable_worker_cache(one,folder)
                write_json(stamp,expected_sig)
            r=_rank_cached_frame(one,folder,nusc,by_lidar)
            results.append(r)
            print(f'  [SCORE] idx={idx} score={r["selection_score"]:.2f} '
                  f'COUNT GT={r["num_gt_visible"]} / PCC={r["num_pred_ours"]} '
                  f'/ LCC={r["num_pred_lcc"]} '
                  f'COUNT-GAP PCC={r["count_gap_ours"]} LCC={r["count_gap_lcc"]} '
                  f'TP PCC={r["tp_ours"]} LCC={r["tp_lcc"]} '
                  f'FP PCC={r["fp_ours"]} LCC={r["fp_lcc"]} '
                  f'qualifies={r["qualifies"]} reason={r["filter_reasons"]}',flush=True)
        except Exception as exc:
            print(f'[AUTO ERROR] sample={idx}: {type(exc).__name__}: {exc}',flush=True)
            failures.append({'sample_idx':idx,'type':type(exc).__name__,'error':str(exc)})
            if not args.auto_skip_errors:
                _write_ranking_csv(root/'ranking_partial.csv',results)
                write_json(root/'errors.json',failures)
                raise
        _write_ranking_csv(root/'ranking_partial.csv',results)
        write_json(root/'errors.json',failures)

    results.sort(key=lambda r:(-int(r['qualifies']),-float(r['selection_score']),
                               -int(r['count_gap_advantage']),
                               -int(r['net_tp_gain']),
                               int(r['count_gap_ours']),int(r['sample_idx'])))
    eligible=[r for r in results if r['qualifies']]
    if not eligible and args.auto_allow_no_gain:
        eligible=results
    selected=[]
    used_scenes=set()
    # Prefer diverse scenes so three adjacent frames are not presented as
    # independent examples; fall back to same-scene only if not enough.
    for unique_pass in (True,False):
        for r in eligible:
            if len(selected)>=args.auto_top_k:
                break
            if r in selected:
                continue
            if unique_pass and r['scene_token'] in used_scenes:
                continue
            selected.append(r)
            used_scenes.add(r['scene_token'])
        if len(selected)>=args.auto_top_k:
            break

    from collections import Counter
    rejection_counts=Counter()
    for r in results:
        if not r['qualifies']:
            rejection_counts.update(x for x in r['filter_reasons'].split(';') if x)
    selected_ids={r['sample_idx']:i+1 for i,r in enumerate(selected)}
    for r in results:
        r['selected_rank']=selected_ids.get(r['sample_idx'],'')
    _write_ranking_csv(root/'ranking_all.csv',results)
    write_json(root/'selection_summary.json',{
        'scan_indices':indices,'ranking_sample_count':len(results),
        'qualifying_sample_count':sum(r['qualifies'] for r in results),
        'failed_sample_count':len(failures),
        'rejection_reason_counts':dict(rejection_counts),
        'selected_sample_indices':[r['sample_idx'] for r in selected],
        'selected_scene_tokens':[r['scene_token'] for r in selected],
        'selection_config':{
            'camera':args.cam_name,'score_threshold':args.score_thr,
            'center_match_distance_m':args.auto_match_distance,
            'min_visible_gt':args.auto_min_visible_gt,
            'max_ours_count_gap':args.auto_max_ours_count_gap,
            'min_lcc_count_gap':args.auto_min_lcc_count_gap,
            'min_net_tp_gain':args.auto_min_net_tp_gain,
            'max_ours_unmatched_fp':args.auto_max_ours_fp,
            'lcc_under_detect_only':args.auto_lcc_under_detect,
            'min_unique_pcc_gt_gain':args.auto_min_gain,
            'max_extra_unmatched_pcc':args.auto_max_extra_fp,
            'selection_score':'3*(TP_pcc-TP_lcc) + 2*(|LCC-GT|-|PCC-GT|) - 1.5*max(FP_pcc-FP_lcc,0)',
            'match_type':'confidence-ordered, same-class greedy LiDAR XY center distance',
            'methodology':'GT used only to RANK cached model predictions, not during inference',
            'limits':'diagnostic proxy matching, NOT official nuScenes AP/NDS/TP/FP/FN',
            'selection_bias':'selected favorable qualitative examples; do not describe as representative or typical',
        },
        'selected':selected,
    })
    print(f'[AUTO] qualifying={sum(r["qualifies"] for r in results)}/{len(results)}, '
          f'selected={len(selected)}',flush=True)
    if not selected:
        print('[AUTO] No frame met the stricter COUNT+GT MATCH constraints. '
              'Inspect ranking_all.csv filter_reasons; increase scan count or '
              'relax --auto-max-ours-count-gap / --auto-min-lcc-count-gap / '
              '--auto-min-net-tp-gain rather than inventing a favorable frame.',flush=True)
        return
    for rank,r in enumerate(selected,1):
        idx=r['sample_idx']
        one=copy.copy(args)
        one.sample_idx=idx
        one.out_dir=str(cache_root/f'sample{idx:05d}')
        print(f'[RENDER] rank={rank} sample={idx} score={r["selection_score"]}',flush=True)
        verify_reusable_worker_cache(one,Path(one.out_dir))
        compose_worker_outputs(one,Path(one.out_dir))
        stem=f'sample{idx:05d}_{args.cam_name}'
        files=[
            f'{stem}_detection_only_3col_GT_LCCNet_PCC.png',
            f'{stem}_comparison_3col_GT_LCCNet_PCC.png',
            f'{stem}_manifest.json',
        ]
        for filename in files:
            src=Path(one.out_dir)/filename
            if not src.is_file():
                raise FileNotFoundError(f'Render selected frame missing: {src}')
            dst=selection_root/f'rank{rank:02d}_{filename}'
            shutil.copy2(src,dst)
            print('  [SELECTED]',dst)
    print('[DONE] All scored samples:',root/'ranking_all.csv',flush=True)
    print('[DONE] Selected qualitative figures:',selection_root,flush=True)


def main():
    args=parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if args.run_worker:
        if args.run_worker=='pcc':
            verify_exists(args.pcc_checkpoint,'PCC checkpoint')
            pcc_worker(args,out)
        else:
            verify_exists(args.official_checkpoint,'Official checkpoint')
            official_worker(args,out)
        return
    if args.auto_rank_only and args.auto_search_count <= 0:
        raise ValueError('--auto-rank-only requires --auto-search-count > 0')
    if args.auto_search_count > 0:
        _auto_sample_scan(args)
        return
    if args.reuse_worker_cache:
        verify_reusable_worker_cache(args,out)
        print('[STEP] Recompose 3-column figures using verified cached predictions',flush=True)
    else:
        verify_exists(args.pcc_checkpoint,'PCC checkpoint')
        verify_exists(args.official_checkpoint,'Official detector checkpoint')
        print('[STEP 1/3] PCC inference: separate BEVFusion process',flush=True)
        subprocess.run(worker_command(args,'pcc'),check=True)
        print('[STEP 2/3] Official + LCCNet inference: separate BEVFusion process',flush=True)
        subprocess.run(worker_command(args,'official'),check=True)
    print('[STEP 3/3] Verify matching input and generate figures',flush=True)
    compose_worker_outputs(args,out)


if __name__=='__main__':
    main()
