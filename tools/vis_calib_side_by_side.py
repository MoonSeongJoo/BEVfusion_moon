#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-quality, raw-image calibration overlays for PCC/LCCNet on nuScenes.

Run from the root of this specific mmdetection3d fork.  This script does NOT
import projects.OfficialBEVFusion (its class collides with PCC BEVFusion).

Output: LCCNet, LGPC Stage 1, RRRF Stage 2 diagnostic overlays.  Every PNG
contains GT (green), BROKEN (red), and recovered (yellow) on ONE camera image.
The underlying LiDAR point IDs and broken extrinsic are identical.

V4 CRITICAL FIX: filter and choose points AFTER the full LiDAR scan is
projected onto the GT camera image. V3 randomly chose 3,000 LiDAR points
BEFORE field-of-view clipping, which left only ~267 visible in some frames.

RRRF Stage 2 is a projection of the composed SE(3) residual prediction;
RRRF actually applies its residual to camera proposal geometry, rather than
re-projecting the entire image. Label accordingly in the manuscript.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)
BGR_GT = (40, 225, 45)           # green
BGR_BROKEN = (35, 45, 245)       # red
BGR_RECOVERED = (0, 225, 255)    # yellow


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ours-config', required=True)
    p.add_argument('--ours-checkpoint', required=True)
    p.add_argument('--lccnet-model-checkpoint', required=True,
                   help='nuScenes-trained LCCNet best_joint_model.pth (NOT official BEVFusion pth)')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument('--cam-name', default='CAM_FRONT', choices=CAMERAS)
    p.add_argument('--auto-search', type=int, default=0,
                   help='evaluate first N val frames, then select the most legible improvement example')
    p.add_argument('--out-dir', default='work_dirs/paper_vis_calib_compare')
    p.add_argument('--max-points', type=int, default=5000,
                   help='max GT-visible LiDAR points to draw (subsample AFTER FOV filtering)')
    p.add_argument('--visual-lidar-source', choices=('keyframe', 'multisweep'),
                   default='keyframe',
                   help='keyframe: raw single scan, recommended for GT validation; '
                        'multisweep: original motion-compensated eval cloud (moving-object ghosts possible)')
    p.add_argument('--minimum-gt-visible', type=int, default=150,
                   help='warn when the unthinned GT image contains fewer than this many points')
    p.add_argument('--z-min', type=float, default=-2.5,
                   help='do not use +0.3 by default: removes road/low objects')
    p.add_argument('--z-max', type=float, default=5.0)
    p.add_argument('--min-distance', type=float, default=1.0)
    p.add_argument('--max-distance', type=float, default=75.0)
    p.add_argument('--max-depth', type=float, default=75.0)
    p.add_argument('--seed', type=int, default=20260811)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--scale', type=int, default=1,
                   help='image upscaling for publication PNG')
    p.add_argument('--opacity', type=float, default=0.85)
    p.add_argument('--fail-on-missing-rrrf', action='store_true')
    p.add_argument('--source-image-only', action='store_true',
                   help='also save the unmarked camera image')
    p.add_argument('--min-paired-points', type=int, default=75,
                   help='minimum number of GT-Broken-Recovered matched visible LiDAR points')
    p.add_argument('--projection-gate-atol', type=float, default=2e-3,
                   help='maximum mismatch between supplied dataset lidar2img and K @ inv(GT C2L)')
    p.add_argument('--paired-only', action='store_true',
                   help='only show points simultaneously visible in GT/Broken/Recovered; sparser but paired')
    p.add_argument('--draw-all-visible', action='store_true',
                   help='deprecated compatibility flag: default already shows all independently visible GT-selected points')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_and_runner(cfg_file, checkpoint, device):
    cfg = Config.fromfile(cfg_file)
    cfg.launcher = 'none'
    cfg.work_dir = str(Path('work_dirs/_paper_visualization_runner'))
    cfg.load_from = None
    cfg.resume = False
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    if 'sampler' in cfg.test_dataloader:
        cfg.test_dataloader.sampler.shuffle = False
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, checkpoint, map_location='cpu')
    runner.model.to(device).eval()
    # Force construction exactly once. Do not call the test runner itself.
    return runner


def load_lccnet(checkpoint_file, device):
    # Only the bare LCCNet, not its OfficialBEVFusion wrapper.
    from projects.LCCNetModern.lccnet import LCCNet
    ckpt = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    args = ckpt.get('args', {})
    if not isinstance(args, dict):
        raise RuntimeError('LCCNet checkpoint["args"] is not a dictionary')
    resolution = ckpt.get('resolution')
    feat_from = int(args.get('use_feat_from', -1))
    depth_scale = float(args.get('depth_scale', -1))
    color_order = args.get('input_color_order')
    if list(resolution or []) != [256, 704]:
        raise RuntimeError(f'LCCNet resolution mismatch: {resolution} != [256,704]')
    if feat_from != 1:
        raise RuntimeError(f'LCCNet checkpoint use_feat_from={feat_from}; expected 1')
    if abs(depth_scale - 50.0) > 1e-5:
        raise RuntimeError(f'LCCNet depth_scale={depth_scale}; expected 50')
    if color_order != 'rgb':
        raise RuntimeError(f'LCCNet checkpoint input_color_order={color_order}; expected rgb')
    net = LCCNet(
        resnet_argv=dict(num_layers=18, pretrained=False, frozen=False),
        image_size=(256, 704), use_feat_from=1, md=4,
        use_reflectance=False, dropout=0.0,
        Action_Func='leakyrelu', attention=False,
    )
    net.load_state_dict(ckpt['model'], strict=True)
    net.to(device).eval().requires_grad_(False)
    print('[LCCNet] epoch=', ckpt.get('epoch', '?'),
          'use_feat_from=', feat_from, 'depth_scale=', depth_scale)
    return net, depth_scale


def get_points(obj):
    return obj.tensor if hasattr(obj, 'tensor') else obj


def get_batch(model, raw):
    # Only call preprocessor once; model(inputs, samples, mode='predict') below
    # already expects preprocessed inputs, not unprocessed raw batch.
    processed = model.data_preprocessor(raw, training=False)
    inp = processed['inputs']
    samples = processed['data_samples']
    img = inp['imgs']
    assert img.shape[0] == 1, f'Expected B=1, got {tuple(img.shape)}'
    sample = samples[0]
    device, dtype = img.device, img.dtype

    def sample_mat(name):
        x = getattr(sample, name, None)
        if x is None:
            raise RuntimeError(f'DataSample missing {name}; check CustomPack3DDetInputs')
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if t.ndim != 3 or t.shape[-2:] != (4, 4):
            raise RuntimeError(f'{name} should be [N,4,4], got {tuple(t.shape)}')
        return t[None]

    gt = sample_mat('camera2lidar')
    broken = sample_mat('broken_camera2lidar')
    K = sample_mat('broken_camera_intrinsics')
    if gt.shape != broken.shape or K.shape != gt.shape:
        raise RuntimeError('Camera calibration tensor shapes do not match')

    metas = sample.metainfo
    ncam = gt.shape[1]
    def aug(name, camera_based):
        value = metas.get(name)
        if value is None:
            print(f'[WARN] {name} missing: using identity; verify projection convention')
            arr = np.eye(4, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        if camera_based:
            if arr.ndim == 2:
                arr = np.repeat(arr[None], ncam, axis=0)
            if arr.shape != (ncam, 4, 4):
                raise RuntimeError(f'{name}: expected ({ncam},4,4), got {arr.shape}')
        else:
            if arr.ndim == 3:
                if not np.allclose(arr, arr[0], atol=1e-5):
                    raise RuntimeError('Nonuniform lidar_aug across cameras unsupported')
                arr = arr[0]
            if arr.shape != (4, 4):
                raise RuntimeError(f'{name}: expected (4,4), got {arr.shape}')
        return torch.from_numpy(arr.copy()).to(device=device, dtype=dtype)

    img_aug = aug('img_aug_matrix', True)
    lidar_aug = aug('lidar_aug_matrix', False)
    points = get_points(inp['points'][0]).to(device)
    return dict(inp=inp, samples=samples, sample=sample, metas=metas,
                imgs=img, points=points, gt=gt, broken=broken,
                K=K, img_aug=img_aug, lidar_aug=lidar_aug)


def predict_pcc(model, dat):
    with torch.no_grad():
        out = model(dat['inp'], dat['samples'], mode='predict')
    if not isinstance(out, (list, tuple)) or len(out) != 1:
        raise RuntimeError(f'Unexpected PCC forward result: {type(out)}')
    return out[0]


def prediction_dict(sample, kind):
    # Stage1 is stored on Det3DDataSample.metainfo;
    # Stage2 is stored in InstanceData.metainfo (transfusion_head.py).
    if kind == 's1':
        return sample.metainfo
    inst = getattr(sample, 'pred_instances_3d', None)
    if inst is None:
        return {}
    return getattr(inst, 'metainfo', {}) or {}


def get_matrix_from_predictor(pred_sample, model, dat):
    mi = prediction_dict(pred_sample, 's1')
    r1 = mi.get('stage1_pred_delta_rot', mi.get('pred_delta_rot_1st'))
    t1 = mi.get('stage1_pred_delta_trans', mi.get('pred_delta_trans_1st'))
    if r1 is None or t1 is None:
        raise RuntimeError('LGPC Stage1 pred_delta missing in Det3DDataSample.metainfo')
    dev, dt = dat['gt'].device, dat['gt'].dtype
    r1 = torch.as_tensor(r1, device=dev, dtype=dt)
    t1 = torch.as_tensor(t1, device=dev, dtype=dt)
    if r1.ndim == 2:
        r1 = r1.unsqueeze(0)
        t1 = t1.unsqueeze(0)
    if r1.shape != dat['gt'].shape[:2] + (3,):
        raise RuntimeError(f'Stage1 delta shape mismatch: {tuple(r1.shape)}')
    stage1 = model._get_corrected_calib_from_prediction(
        r1, t1, dat['broken'], dat['K'])['cam2lidar']

    active = mi.get('stage1_active_cam_indices')
    if active is not None:
        active = torch.as_tensor(active).long().flatten().tolist()
    mi2 = prediction_dict(pred_sample, 's2')
    # Head predicts R_total = R1 @ R2, t_total = t1 + R1 @ t2;
    # same convention as PCC metric: inv(Delta_total) @ T_broken.
    rt = mi2.get('pred_delta_rot_total_R')
    tt = mi2.get('pred_delta_trans_total')
    stage2 = None
    if rt is not None and tt is not None:
        rt = torch.as_tensor(rt, dtype=dt, device=dev)
        tt = torch.as_tensor(tt, dtype=dt, device=dev)
        if rt.ndim == 4:
            if rt.shape[0] != 1:
                raise RuntimeError('Stage2 unexpected batch dimension')
            rt = rt[0]
        if tt.ndim == 3:
            if tt.shape[0] != 1:
                raise RuntimeError('Stage2 unexpected batch dimension')
            tt = tt[0]
        ncam = dat['gt'].shape[1]
        if rt.shape != (ncam, 3, 3) or tt.shape != (ncam, 3):
            raise RuntimeError(f'Stage2 total shapes: {tuple(rt.shape)}, {tuple(tt.shape)}')
        delta_total = torch.eye(4, dtype=dt, device=dev).expand(ncam, 4, 4).clone()
        delta_total[:, :3, :3] = rt
        delta_total[:, :3, 3] = tt
        stage2 = (torch.linalg.inv(delta_total)[None] @ dat['broken'])
    else:
        print('[RRRF] Stage2 total SE(3) keys absent in pred_instances_3d.metainfo.')
        print('       keys:', sorted(mi2.keys()))
    return stage1, stage2, active


def predict_lccnet(lccnet, depth_scale, model, dat):
    from projects.LCCNetModern.lie import se3
    from projects.LCCNetModern.calib_geometry import project_lidar_to_sparse_depth
    img = dat['imgs']
    B, N, C, H, W = img.shape
    if (H, W) != (256, 704):
        raise RuntimeError(f'LCCNet requires 256x704, got {(H, W)}')
    bcal = model._build_calib_dict_from_cam2lidar(dat['broken'], dat['K'])
    depth = project_lidar_to_sparse_depth(
        points=[dat['points']], lidar2image=bcal['lidar2img'],
        img_aug_matrix=dat['img_aug'][None],
        lidar_aug_matrix=dat['lidar_aug'][None],
        image_hw=(H, W),
    ) / depth_scale
    # Match OfficialBEVFusion._bevfusion_img_to_lccnet, not generic ImageNet
    # preprocessing.  This mirrors the user-trained checkpoint's input path.
    x = img.float()
    if float(x.min()) >= 0.0 and float(x.max()) > 2.0:
        x = x / 255.0
    m = x.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
    s = x.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    rgb = (x - m) / s
    with torch.no_grad():
        pred = lccnet(rgb.reshape(B*N, C, H, W).float(),
                      depth.reshape(B*N, 1, H, W).float())
    if pred.shape != (B*N, 6) or not torch.isfinite(pred).all():
        raise RuntimeError(f'LCCNet output invalid: {tuple(pred.shape)}')
    corr = se3.exp(pred).reshape(B, N, 4, 4)
    return corr @ dat['broken']


def resolve_input_path(name, known_roots=None):
    """Resolve original nuScenes paths from metainfo; do not guess wrong camera frames."""
    path = Path(str(name))
    roots = [Path.cwd(), Path('data/nuscenes'), Path('/workspace/mmdetection3d'),
             Path('/workspace/mmdetection3d/data/nuscenes')]
    if known_roots:
        roots = list(known_roots) + roots
    if path.is_file():
        return path
    if not path.is_absolute():
        for root in roots:
            candidate = root / path
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f'DataSample source path not found: {name}')


def _range_filter_xyz(xyz, args):
    """Filter full RAW-frame point cloud without FOV subsampling."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise RuntimeError(f'Expected LiDAR Nx3(+), got {xyz.shape}')
    xyz = xyz[:, :3]
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    keep = (np.isfinite(xyz).all(axis=1)
            & (xyz[:, 2] >= args.z_min)
            & (xyz[:, 2] <= args.z_max)
            & (radius >= args.min_distance)
            & (radius <= args.max_distance))
    return xyz[keep]


def visualization_points(dat, args):
    """Return all eligible LiDAR points; do NOT subsample before the GT FOV.

    keyframe: read current raw LIDAR_TOP .bin (5 floats / point).
    multisweep: point tensor used by the evaluation pipeline but restore the
        original LiDAR coordinate frame (PointToMultiViewDepth already builds
        points_original by undoing lidar augmentation).
    """
    if args.visual_lidar_source == 'keyframe':
        name = dat['metas'].get('lidar_path')
        if not name:
            raise RuntimeError('DataSample lacks lidar_path; cannot read current raw LiDAR keyframe')
        path = resolve_input_path(name)
        if path.suffix == '.npy':
            arr = np.load(str(path))
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise RuntimeError(f'Invalid .npy raw LiDAR shape {arr.shape}')
            xyz = np.asarray(arr[:, :3], dtype=np.float32)
        else:
            from mmengine.fileio import get
            payload = np.frombuffer(get(str(path)), dtype=np.float32)
            if payload.size == 0 or payload.size % 5 != 0:
                raise RuntimeError(f'nuScenes keyframe LiDAR expected Nx5 floats: {path}')
            xyz = payload.reshape(-1, 5)[:, :3].copy()
        label = 'keyframe'
        source = str(path)
    else:
        packed = dat['inp'].get('points_original')
        if packed is None:
            raise RuntimeError('Multi-sweep visualization requires packed points_original; '
                               'do not substitute augmented points without undoing lidar_aug_matrix')
        if isinstance(packed, (tuple, list)):
            if len(packed) != 1:
                raise RuntimeError(f'Expected B=1 points_original, got B={len(packed)}')
            packed = packed[0]
        packed = get_points(packed)
        if not hasattr(packed, 'detach'):
            packed = torch.as_tensor(packed)
        xyz = packed.detach().cpu().numpy()[:, :3].copy()
        label = 'multisweep (dynamic-object ghosts possible)'
        source = 'evaluation points_original (multi-sweep, original LiDAR frame)'
    raw_count = len(xyz)
    xyz = _range_filter_xyz(xyz, args)
    if len(xyz) < 75:
        raise RuntimeError(f'Only {len(xyz)} points after physical range filter; original={raw_count}')
    points = torch.as_tensor(xyz, device=dat['gt'].device, dtype=dat['gt'].dtype)
    return points, source, raw_count, label


def choose_gt_visible_points(xyz, gt_c2l, K, image_hw, args, frame_idx):
    """Crucial ordering: full scan -> project GT -> FOV -> optional sample.

    The same resulting IDs are then projected using GT, Broken, LCCNet,
    LGPC, RRRF without any method-specific resampling or cherry-picking.
    """
    _uv, gt_mask = project_raw_keyframe(xyz, gt_c2l, K,
                                        image_hw, args.max_depth)
    eligible_ids = np.flatnonzero(gt_mask)
    full_count = int(eligible_ids.size)
    if full_count == 0:
        raise RuntimeError('GT projection has zero visible LiDAR returns. Check calibration/coordinate conventions.')
    if full_count < args.minimum_gt_visible:
        print(f'[WARN] Only {full_count} GT-visible points BEFORE subsampling. '
              'Check camera, range limits and LiDAR synchronization.')
    limit = args.max_points
    if full_count > limit:
        # Stable and reproducible; no favored method participates in selection.
        rng = np.random.default_rng(int(args.seed) + int(frame_idx))
        chosen = np.sort(rng.choice(eligible_ids, size=limit, replace=False))
    else:
        chosen = eligible_ids
    idx = torch.as_tensor(chosen, device=xyz.device, dtype=torch.long)
    return xyz.index_select(0, idx), full_count


def verify_clean_projection(dat, cam_idx, atol):
    """Mandatory numeric gate independent of our projection implementation.

    Project metadata path: BEVLoadMultiViewImageFromFiles stores lidar2img
    K_original @ lidar2cam (before ImageAug3D).  PointToMultiViewDepth saves
    ground-truth c2l from the same lidar2cam.  They must agree.
    """
    meta = dat['metas']
    if 'lidar2img' not in meta:
        raise RuntimeError('Dataset clean lidar2img not in metainfo: cannot certify GT reference.')
    gt = dat['gt'][0, cam_idx].detach().cpu().double()
    K = dat['K'][0, cam_idx].detach().cpu().double()
    P_calc = (K @ torch.linalg.inv(gt))[:3, :]
    P_meta = torch.as_tensor(meta['lidar2img'][cam_idx]).detach().cpu().double()[:3, :]
    p_error = float((P_calc - P_meta).abs().max())
    print(f'[GT PROJECTION GATE] K @ inv(GT) vs dataset lidar2img: {p_error:.6g} (threshold {atol})')
    if p_error > atol or not np.isfinite(p_error):
        raise RuntimeError('GT projection matrix mismatch. Stop before publishing any overlay.')
    B = dat['broken'][0, cam_idx].detach().cpu().double()
    D = B @ torch.linalg.inv(gt)
    if torch.linalg.norm(D[:3, :3] - torch.eye(3, dtype=D.dtype)) < 1e-8 and torch.linalg.norm(D[:3, 3]) < 1e-8:
        raise RuntimeError('BROKEN == GT for selected camera: perturbation not applied')
    return p_error


def project_raw_keyframe(xyz_raw, c2l, K, hw, max_depth):
    """Raw LiDAR_xyz -> Raw camera pixels, with no lidar_aug or img_aug.

    This deliberately matches the BEFORE-augmentation projection in
    PointToMultiViewDepth: P=cam2img @ lidar2cam.  The camera background is
    loaded from original .jpg at its ORIGINAL resolution.
    """
    H, W = hw
    xyz = xyz_raw.double()
    T = c2l.double().to(xyz.device)
    intrinsic = K.double().to(xyz.device)
    ones = torch.ones_like(xyz[:, :1])
    points_h = torch.cat([xyz, ones], dim=-1)
    l2c = torch.linalg.inv(T)
    xyz_cam = (points_h @ l2c.T)[:, :3]
    depth = xyz_cam[:, 2]
    uvw = (xyz_cam @ intrinsic[:3, :3].T)
    uv = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-7)
    mask = (torch.isfinite(uv).all(dim=1)
            & (depth > 1.0) & (depth < max_depth)
            & (uv[:, 0] >= 0) & (uv[:, 0] < W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    return uv.float().cpu().numpy(), mask.cpu().numpy().astype(bool)


def camera_path(metas, cam_idx, cam_name):
    paths = metas.get('img_path')
    if isinstance(paths, (tuple, list)) and len(paths) > cam_idx:
        chosen = str(paths[cam_idx])
        # A mismatch here means the hardcoded camera order is wrong, making
        # all qualitative conclusions unreliable.
        if 'CAM_' in chosen and cam_name not in chosen:
            raise RuntimeError(
                f'Camera order mismatch: requested {cam_name} but img_path={chosen}')
        return chosen
    if isinstance(paths, dict):
        return paths.get(cam_name)
    raise RuntimeError('img_path is missing from DataSample.metainfo')


def read_camera_bgr(dat, cam_idx, cam_name):
    """Read ORIGINAL nuScenes image; do NOT resize/warp it.

    Clean K and GT extrinsic are ORIGINAL camera coordinates. Mixing this
    image with post-augmentation coordinates (or vice versa) breaks overlay.
    """
    name = camera_path(dat['metas'], cam_idx, cam_name)
    path = resolve_input_path(name)
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f'cv2.imread failed: {path}')
    return raw, str(path)


def geom_metrics(uvs):
    gt, vb = uvs['GT']
    metrics = {}
    for method in ('Broken', 'LCCNet', 'LGPC', 'RRRF'):
        if method not in uvs:
            continue
        xy, vm = uvs[method]
        common = vb & vm
        n = int(common.sum())
        median = float(np.median(np.linalg.norm(xy[common]-gt[common], axis=1))) if n else None
        mean = float(np.mean(np.linalg.norm(xy[common]-gt[common], axis=1))) if n else None
        metrics[method] = dict(common_visible=n, pixel_error_mean=mean,
                               pixel_error_median=median)
    # A **paired**, intersection-on-all-methods diagnostic for selecting frames:
    same = vb.copy()
    for _uv, visible in uvs.values():
        same &= visible
    pairs = {}
    if int(same.sum()) > 0:
        for k, (uv, _) in uvs.items():
            pairs[k] = float(np.median(np.linalg.norm(uv[same]-gt[same], axis=1)))
    return metrics, int(same.sum()), pairs


def evaluate_one(model, lccnet, depth_scale, raw_batch, args, cam_idx, sample_idx):
    dat = get_batch(model, raw_batch)
    if dat['gt'].shape[1] <= cam_idx:
        raise RuntimeError('Insufficient cameras in DataSample')
    # Check GT -> dataset camera matrix BEFORE using model outputs.
    matrix_error = verify_clean_projection(dat, cam_idx, args.projection_gate_atol)
    # Input network still receives augmented images and multi-sweep input as
    # trained. Visualization uses independent raw single-scan keyframe points.
    result = predict_pcc(model, dat)
    s1, s2, active = get_matrix_from_predictor(result, model, dat)
    if active is not None and cam_idx not in active:
        print(f'[WARN] {args.cam_name} not in active LGPC cameras={active}; do not claim Stage1 recovery for this view.')
    lcc = predict_lccnet(lccnet, depth_scale, model, dat)
    xyz_full, lidar_path, raw_count, source_label = visualization_points(dat, args)
    cam_bgr, photo_path = read_camera_bgr(dat, cam_idx, args.cam_name)
    H, W = cam_bgr.shape[:2]
    K = dat['K'][0, cam_idx]
    xyz, gt_visible_full = choose_gt_visible_points(
        xyz_full, dat['gt'][0, cam_idx], K, (H, W), args, sample_idx)
    print(f'[POINT-GATE {sample_idx}] source={source_label} raw={raw_count} '
          f'range_filtered={len(xyz_full)} gt_visible_before_limit={gt_visible_full} '
          f'gt_visible_after_limit={len(xyz)} max_points={args.max_points}')
    transforms = {
        'GT': dat['gt'][0, cam_idx],
        'Broken': dat['broken'][0, cam_idx],
        'LCCNet': lcc[0, cam_idx],
        'LGPC': s1[0, cam_idx],
    }
    if s2 is not None:
        transforms['RRRF'] = s2[0, cam_idx]
    uvs = {name: project_raw_keyframe(xyz, T, K, (H, W), args.max_depth)
           for name, T in transforms.items()}
    gt_count = int(uvs['GT'][1].sum())
    if gt_count < 30:
        raise RuntimeError(f'Only {gt_count} GT points visible on {args.cam_name} '
                           'after FOV selection; inspect projection convention.')
    metrics, common_count, paired = geom_metrics(uvs)
    broken_err = paired.get('Broken', 0.0)
    lgpc_err = paired.get('LGPC', float('inf'))
    score = 0.0
    # Select scenes with a reasonable number of EXACTLY IDENTICAL points
    # visible in every method; avoid closest-neighbor shortcut.
    if common_count >= args.min_paired_points and broken_err >= 4.0:
        gain = max(0., (broken_err - lgpc_err) / max(1., broken_err))
        score = (min(common_count, 2500) ** .5) * gain * np.log1p(broken_err)
    print('[IN-VIEW]', ' '.join(f'{name}={int(mask.sum())}' for name, (_, mask) in uvs.items()))
    print(f'[FRAME {sample_idx}] raw={raw_count} range_filtered={len(xyz_full)} '
          f'gt_visible_full={gt_visible_full} drawn_candidates={len(xyz)} '
          f'common={common_count} broken={broken_err:.2f}px '
          f'LGPC={lgpc_err:.2f}px score={score:.3f} active={active} RRRF={s2 is not None}')
    return dict(idx=sample_idx, score=float(score), image=cam_bgr,
                photo=photo_path, lidar_path=lidar_path, gt_proj_gate=matrix_error,
                uvs=uvs, metrics=metrics, common_count=common_count,
                paired=paired, active=active, total_points=len(xyz),
                raw_lidar_points=raw_count, gt_visible=gt_count,
                gt_visible_full=gt_visible_full, range_filtered_points=len(xyz_full),
                point_source=source_label, cam=args.cam_name)


def select_one(runner, model, lccnet, depth_scale, args, cam_idx):
    loader = runner.test_dataloader
    if args.auto_search > 0:
        results = []
        for idx, raw in enumerate(loader):
            if idx >= args.auto_search:
                break
            # Fail fast on format or implementation errors: don't hide code
            # bugs as if some frames were simply unsuitable.
            candidate = evaluate_one(model, lccnet, depth_scale, raw,
                                     args, cam_idx, idx)
            results.append(candidate)
        if not results:
            raise RuntimeError('Empty test dataloader')
        ranked = sorted(results, key=lambda r: r['score'], reverse=True)
        best = ranked[0]
        if best['score'] == 0:
            print('[WARN] None of the searched samples passed the visibility/improvement gate; '
                  'first candidate selected. Inspect before publication.')
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir/'sample_selection.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_idx', 'score', 'paired_common_points',
                             'broken_px_median', 'lgpc_px_median'])
            for r in ranked:
                writer.writerow([r['idx'], f"{r['score']:.6f}",r['common_count'],
                                 r['paired'].get('Broken'),r['paired'].get('LGPC')])
        print(f'[SELECTED] index={best["idx"]} (searched {len(results)} frames)')
        return best
    for idx, raw in enumerate(loader):
        if idx == args.sample_idx:
            return evaluate_one(model, lccnet, depth_scale, raw,
                                args, cam_idx, idx)
    raise IndexError(f'sample_idx={args.sample_idx} outside test dataloader')


def draw_rgb_legend(image, title, note=None, scale=2, entries=None):
    """Place label banner OUTSIDE image, not over objects/points."""
    h, w = image.shape[:2]
    s = max(1, int(scale))
    big = cv2.resize(image, (w*s, h*s), interpolation=cv2.INTER_CUBIC)
    banner_h = 90*s
    panel = np.zeros((banner_h + big.shape[0], big.shape[1], 3), dtype=np.uint8)
    panel[:banner_h] = (31, 34, 35)
    panel[banner_h:] = big
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, s*0.43)
    cv2.putText(panel, title, (14*s, 20*s), font, fs*1.05,
                (255, 255, 255), max(1, s), cv2.LINE_AA)
    if entries is None:
        entries = [('GT', BGR_GT), ('Broken', BGR_BROKEN), ('Recovered', BGR_RECOVERED)]
    x0 = 15*s
    y = 41*s
    for label, color in entries:
        cv2.circle(panel, (x0+5*s, y), max(2,s*2), color, -1, cv2.LINE_AA)
        cv2.putText(panel, label, (x0+15*s, y+4*s), font,
                    fs, (240,240,240), max(1,s), cv2.LINE_AA)
        x0 += {'GT':93,'Broken':138,'Recovered':0}[label] * s
    if note:
        cv2.putText(panel, note[:85], (15*s, 65*s), font,
                    fs*.64, (195, 205, 213), max(1,s//2), cv2.LINE_AA)
    return panel


def render_overlay(frame, method, args):
    base = frame['image'].copy()
    uv_gt, m_gt = frame['uvs']['GT']
    uv_br, m_br = frame['uvs']['Broken']
    uv_rc, m_rc = frame['uvs'][method]
    # Rendering paired physical point IDs is critical: differences between
    # all three projections refer to the SAME LiDAR returns.
    common = m_gt & m_br & m_rc
    if args.paired_only:
        if int(common.sum()) < 25:
            print(f'[WARN] {method}: only {common.sum()} jointly visible points in paired-only mode')
        masks = (common, common, common)
    else:
        # Same set of source point IDs, but an off-image projection of one
        # method must not hide a valid GT return for the other methods.
        masks = (m_br, m_gt, m_rc)
    alpha = max(0., min(1., args.opacity))
    layer = base.copy()
    for uv, mask, color in ((uv_br, masks[0], BGR_BROKEN),
                            (uv_gt, masks[1], BGR_GT),
                            (uv_rc, masks[2], BGR_RECOVERED)):
        for x, y in np.rint(uv[mask]).astype(np.int32):
            cv2.circle(layer, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    base = cv2.addWeighted(layer, alpha, base, 1-alpha, 0)
    label = {
        'LCCNet': 'LCCNet calibration correction',
        'LGPC': 'Ours: LGPC Stage-1 correction',
        'RRRF': 'Ours: RRRF Stage-2 residual (diagnostic)',
    }[method]
    note = (f"nuScenes {frame['cam']} | val sample {frame['idx']} | "
            f"{frame['point_source']} | GT {frame['gt_visible']} | matched {int(common.sum())}")
    return draw_rgb_legend(base, label, note, args.scale)


def render_gt_only(frame, args):
    base = frame['image'].copy()
    uv, mask = frame['uvs']['GT']
    layer = base.copy()
    for x, y in np.rint(uv[mask]).astype(np.int32):
        cv2.circle(layer, (int(x), int(y)), 2, BGR_GT, -1, cv2.LINE_AA)
    base = cv2.addWeighted(layer, args.opacity, base, 1.-args.opacity, 0)
    # The FIRST visual gate: user must verify walls, road and cars align.
    return draw_rgb_legend(base, 'GT clean extrinsic - VERIFY THIS FIRST',
                           f"{frame['point_source']} | GT in view: {frame['gt_visible_full']} (shown {frame['gt_visible']})",
                           args.scale, entries=[('GT', BGR_GT)])


def write_outputs(frame, args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"sample{frame['idx']:05d}_{frame['cam']}"
    # ALWAYS save mandatory clean GT visual gate and unmodified source.
    cv2.imwrite(str(out/f'{stem}_camera_original.png'), frame['image'])
    cv2.imwrite(str(out/f'{stem}_GT_ONLY_CHECK_FIRST.png'), render_gt_only(frame,args))
    print('[GATE] First inspect', out/f'{stem}_GT_ONLY_CHECK_FIRST.png')
    outputs = []
    for method in ('LCCNet', 'LGPC', 'RRRF'):
        if method not in frame['uvs']:
            print(f'[SKIP] {method} has no valid SE(3) prediction')
            continue
        fname = out / f'{stem}_{method.lower()}_gt_broken_recovered.png'
        image = render_overlay(frame, method, args)
        if not cv2.imwrite(str(fname), image):
            raise RuntimeError(f'Could not save: {fname}')
        outputs.append((method, str(fname), image))
        print('[SAVE]', fname)
    if args.fail_on_missing_rrrf and 'RRRF' not in frame['uvs']:
        raise RuntimeError('RRRF Stage2 SE(3) absent; cannot produce requested third image')
    # Contact sheet: identical camera image / identical points across columns.
    if outputs:
        h = max(x[2].shape[0] for x in outputs)
        panels = []
        for method, name, im in outputs:
            if im.shape[0] < h:
                im = cv2.copyMakeBorder(im, 0, h-im.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0,0,0))
            panels.append(im)
        contact = np.concatenate(panels, axis=1)
        cv2.imwrite(str(out/f'{stem}_paper_contact_sheet.png'), contact)
        print('[SAVE]', out/f'{stem}_paper_contact_sheet.png')
    manifest = dict(
        selected_idx=frame['idx'], camera=frame['cam'], source_image=frame['photo'],
        gt_color='green', broken_color='red', recovered_color='yellow',
        source_lidar_points=frame['total_points'], original_scan_points=frame['raw_lidar_points'], raw_lidar_path=frame['lidar_path'], gt_projection_matrix_error=frame['gt_proj_gate'], active_stage1_cameras=frame['active'],
        range_filtered_points=frame['range_filtered_points'],
        gt_visible_before_fov_downsample=frame['gt_visible_full'],
        visual_lidar_source=frame['point_source'],
        metrics=frame['metrics'], paired_median_errors=frame['paired'],
        same_visible_lidar_points=frame['common_count'],
        sample_selection_score=frame['score'],
        nominal_injection_range='per-axis +/-10 deg and +/-0.75 m (verify actual dataset config)',
        projection_coordinate_frame='RAW camera pixels and RAW LiDAR frame. No ImageAug and no LiDARAug during visualization. For multisweep, prior sweeps are aligned to current LiDAR frame by dataset pipeline.',
        gt_visual_gate='Must visually verify GT_ONLY_CHECK_FIRST.png before claiming correct alignment',
        stage2_note='Composed RRRF SE(3) residual is a diagnostic camera-proposal correction, not a second full image reprojection.',
        generated=[n for _m,n,_x in outputs],
    )
    with (out/f'{stem}_manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('[IMPORTANT] Numeric GT matrix gate passes, but MUST visually check GT_ONLY_CHECK_FIRST.png first.')
    print(f"[INFO] Projection source: {frame['point_source']}; diagnostics, not mAP/NDS evidence.")


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.max_points < 100:
        raise ValueError('--max-points must be >= 100')
    if args.min_paired_points < 0:
        raise ValueError('--min-paired-points must be nonnegative')
    cam_idx = CAMERAS.index(args.cam_name)
    runner = config_and_runner(args.ours_config, args.ours_checkpoint, args.device)
    ours_model = runner.model
    lccnet, depth_scale = load_lccnet(args.lccnet_model_checkpoint, args.device)
    try:
        result = select_one(runner, ours_model, lccnet, depth_scale, args, cam_idx)
        write_outputs(result, args)
    finally:
        del lccnet
    print('[DONE] Review images + manifest; figures are diagnostics, not mAP/NDS evidence.')


if __name__ == '__main__':
    main()
