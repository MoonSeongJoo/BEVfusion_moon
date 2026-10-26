"""Deterministic scene-aware LiDAR-camera extrinsic drift schedules for PCC/nuScenes.

This standalone utility does NOT change the user's BEVFusion code. It builds a
sample-token+camera indexed manifest that can be consumed at the existing
PointToMultiViewDepth broken-camera2lidar injection point.

Convention: T_broken = Delta_T(t) @ T_GT_camera2lidar.
Units: rotation magnitude in degrees; translation magnitude in metres.
Dependencies for generation: numpy. NuScenes devkit is optional, but is needed
if the infos pickle does not already contain scene_token metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
)


def _stable_rng(*parts: Any) -> np.random.Generator:
    key = '|'.join(map(str, parts)).encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    return np.random.default_rng(seed)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=3)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else np.array([1., 0., 0.])


def se3_from_rotvec_translation(rotvec_rad: Iterable[float], translation_m: Iterable[float]) -> np.ndarray:
    """Rodrigues + direct translation; left perturbation in LiDAR frame.

    This is the SE(3) homogeneous transform [R,t;0,1], NOT exp(se3_twist)
    with a coupled V(rho) translation parameterization.
    """
    w = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    t = np.asarray(translation_m, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    wx, wy, wz = w
    W = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=np.float64)
    if theta < 1e-10:
        R = np.eye(3) + W + 0.5 * (W @ W)
    else:
        R = np.eye(3) + np.sin(theta)/theta * W + (1.0-np.cos(theta))/(theta**2) * (W @ W)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _amplitude(mode: str, time_fraction: float,
               shock_fraction: float, return_fraction: float) -> float:
    # Time-based normalized progress, not list-index progress: robust to
    # non-uniform timestamps and dropped nuScenes keyframes.
    p = time_fraction
    if mode == 'fixed':
        return 1.0
    if mode == 'temporal':
        return p  # linear progressive drift from 0 to selected magnitude
    if mode == 'shock':
        return float(p >= shock_fraction)
    if mode == 'shock_return':
        return float(shock_fraction <= p < return_fraction)
    raise ValueError(f'Unknown scenario: {mode}')


def _timestamp_to_us_from_info(raw_timestamp: Any) -> int:
    """Parse nuScenes-like info timestamps without losing subsecond precision.

    Official nuScenes sample timestamps are integer microseconds (~1e15).
    Some converted info files store Unix seconds as float (~1e9).
    A whole-second integer value has already lost keyframe precision, so it is
    intentionally rejected: provide --nuscenes-root to recover canonical time.
    """
    if isinstance(raw_timestamp, np.generic):
        raw_timestamp = raw_timestamp.item()
    if isinstance(raw_timestamp, bool):
        raise ValueError('Boolean timestamp is not valid.')
    if isinstance(raw_timestamp, int):
        if 10**14 <= abs(raw_timestamp) < 10**17:
            return int(raw_timestamp)
        if 10**9 <= abs(raw_timestamp) < 10**11:
            raise ValueError(
                f'Timestamp {raw_timestamp} is whole Unix seconds; '
                'subsecond precision has been lost. Pass --nuscenes-root '
                'to recover official nuScenes sample timestamps.'
            )
    if isinstance(raw_timestamp, (float, int)):
        val = float(raw_timestamp)
        if not np.isfinite(val):
            raise ValueError(f'Non-finite timestamp: {raw_timestamp}')
        if 10**14 <= abs(val) < 10**17:
            return int(round(val))
        if 10**9 <= abs(val) < 10**11:
            return int(round(val * 1_000_000))
    raise ValueError(
        f'Unknown nuScenes timestamp unit/value: {raw_timestamp!r}. '
        'Pass --nuscenes-root to use authoritative nuScenes sample records.'
    )


def extract_scene_frames(infos_path: str, nuscenes_root: str | None = None,
                         nuscenes_version: str = 'v1.0-trainval') -> dict[str, list[dict]]:
    """Return token-keyed scene frames ordered by canonical sample timestamps.

    IMPORTANT: With --nuscenes-root, always get both scene and timestamp from
    NuScenes.get('sample', token), even if converted infos already have fields.
    Never remove duplicate timestamps or drop samples silently.
    """
    with open(infos_path, 'rb') as f:
        obj = pickle.load(f)
    rows = obj.get('infos', obj.get('data_list')) if isinstance(obj, dict) else obj
    if not isinstance(rows, (list, tuple)):
        raise TypeError('Expected infos list, dict[infos], or dict[data_list].')

    nusc = None
    if nuscenes_root:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=nuscenes_version,
                       dataroot=nuscenes_root, verbose=False)

    scene_frames: dict[str, list[dict]] = defaultdict(list)
    seen_tokens = set()
    for i, info in enumerate(rows):
        token = info.get('token') or info.get('sample_token') or info.get('sample_idx')
        if not isinstance(token, str) or len(token) < 20:
            raise ValueError(f'Row {i}: missing nuScenes sample token: {token!r}.')
        if token in seen_tokens:
            raise ValueError(f'Row {i}: duplicate sample token in dataset infos: {token}')
        seen_tokens.add(token)

        if nusc is not None:
            # Canonical nuScenes keyframe timestamp is in microseconds.
            try:
                record = nusc.get('sample', token)
            except Exception as exc:
                raise ValueError(
                    f'Row {i}: sample token={token} is not in '
                    f'nuScenes version={nuscenes_version}; check --nuscenes-root/version.'
                ) from exc
            scene = str(record['scene_token'])
            timestamp_us = int(record['timestamp'])
            info_scene = info.get('scene_token')
            if info_scene is not None and str(info_scene) != scene:
                raise ValueError(
                    f'Row {i}, token={token}: scene mismatch between infos '
                    f'({info_scene}) and nuScenes API ({scene}).'
                )
        else:
            scene = info.get('scene_token')
            raw_timestamp = info.get('timestamp')
            if scene is None or raw_timestamp is None:
                raise ValueError(
                    f'Row {i}: missing scene_token/timestamp. '
                    'Pass --nuscenes-root to query nuScenes API.'
                )
            scene = str(scene)
            timestamp_us = _timestamp_to_us_from_info(raw_timestamp)

        scene_frames[scene].append({
            'token': token,
            'timestamp': timestamp_us,
        })

    for scene, frames in scene_frames.items():
        frames.sort(key=lambda x: (x['timestamp'], x['token']))
        previous = None
        for frame in frames:
            if previous is not None and previous['timestamp'] == frame['timestamp']:
                raise ValueError(
                    'Duplicate timestamps remain within one scene after '
                    'canonical nuScenes lookup: '
                    f'scene={scene}, timestamp_us={frame["timestamp"]}, '
                    f'tokens={previous["token"]}, {frame["token"]}. '
                    'Inspect the nuScenes records and dataset infos; '
                    'do not silently deduplicate keyframes.'
                )
            previous = frame

    if sum(map(len, scene_frames.values())) != len(rows):
        raise AssertionError('Sample count changed during scene grouping.')
    return dict(scene_frames)


def make_schedule(scene_frames: dict[str, list[dict]], *, scenario: str,
                  rot_deg: float, trans_m: float, seed: int = 2027,
                  cameras: Iterable[str] = CAMERAS,
                  shock_fraction: float = 0.50,
                  return_fraction: float = 0.75) -> dict:
    if not (0 <= shock_fraction < return_fraction <= 1):
        raise ValueError('Require 0 <= shock_fraction < return_fraction <= 1')
    if rot_deg < 0 or trans_m < 0:
        raise ValueError('Negative error magnitude is invalid.')
    if scenario not in ('fixed', 'temporal', 'shock', 'shock_return'):
        raise ValueError(f'Unsupported scenario {scenario}')
    cams = tuple(cameras)
    if any(c not in CAMERAS for c in cams) or not cams:
        raise ValueError('cameras must be a nonempty subset of the six nuScenes cameras')

    samples = {}
    for scene_id in sorted(scene_frames):
        frames = scene_frames[scene_id]
        direction = {}
        for cam in cams:
            rng = _stable_rng(seed, scene_id, cam)
            direction[cam] = (_unit_vector(rng), _unit_vector(rng))
        for i, frame in enumerate(frames):
            sample_token = frame['token']
            start_us = frames[0]['timestamp']
            end_us = frames[-1]['timestamp']
            progress = (frame['timestamp'] - start_us) / max(1, end_us - start_us)
            alpha = _amplitude(scenario, progress, shock_fraction, return_fraction)
            samples[sample_token] = {}
            for cam in CAMERAS:
                if cam in direction:
                    rhat, that = direction[cam]
                    rv = rhat * np.deg2rad(rot_deg * alpha)
                    tr = that * trans_m * alpha
                else:
                    rv = np.zeros(3)
                    tr = np.zeros(3)
                T = se3_from_rotvec_translation(rv, tr)
                samples[sample_token][cam] = {
                    'delta_T': T.tolist(),
                    'rotvec_rad': rv.tolist(),
                    'translation_m': tr.tolist(),
                    'rot_deg': float(np.rad2deg(np.linalg.norm(rv))),
                    'trans_m': float(np.linalg.norm(tr)),
                    'scene_token': scene_id,
                    'timestamp': int(frame['timestamp']),
                    'frame_index': i,
                    'scene_length': len(frames),
                    'drift_alpha': alpha,
                    'scene_time_fraction': progress,
                }
    out = {
        'protocol_version': 'PCC-temporal-drift-v1',
        'convention': 'T_broken = Delta_T @ T_GT_camera2lidar (left-multiply in LiDAR frame)',
        'scenario': scenario,
        'seed': seed,
        'rot_magnitude_deg': rot_deg,
        'translation_magnitude_m': trans_m,
        'affected_cameras': list(cams),
        'shock_fraction': shock_fraction,
        'return_fraction': return_fraction,
        'scene_count': len(scene_frames),
        'sample_count': len(samples),
        'samples': samples,
    }
    digest = hashlib.sha256(json.dumps(out, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    out['manifest_sha256'] = digest
    return out


class DriftLookup:
    """Worker-safe immutable lookup; load in dataset/transform __init__ once."""
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        self.samples = self.manifest['samples']

    def get(self, sample_token: str, camera_name: str) -> np.ndarray:
        if sample_token not in self.samples:
            raise KeyError(f'Sample {sample_token!r} missing from drift manifest')
        if camera_name not in self.samples[sample_token]:
            raise KeyError(f'Camera {camera_name!r} missing for sample {sample_token}')
        return np.asarray(self.samples[sample_token][camera_name]['delta_T'], dtype=np.float64)


def make_broken_matrices(gt_cam2lidar: np.ndarray, sample_token: str,
                         cam_names: Iterable[str], schedule: DriftLookup) -> tuple[np.ndarray, np.ndarray]:
    """Return (original, broken) arrays with shape [N,4,4] (does NOT mutate input)."""
    clean = np.asarray(gt_cam2lidar, dtype=np.float64).copy()
    names = tuple(cam_names)
    if clean.shape != (len(names), 4, 4):
        raise ValueError(f'Expected [{len(names)},4,4], got {clean.shape}')
    broken = np.stack([schedule.get(sample_token, cam) @ clean[i]
                       for i, cam in enumerate(names)], axis=0)
    return clean, broken


def _self_test() -> None:
    # Fake tokens and scenes, so testing does not require any nuScenes data.
    scene = {'scene_demo': [{'token': f'{i:032x}', 'timestamp': i * 500000} for i in range(9)]}
    params = dict(rot_deg=10.0, trans_m=0.75, cameras=['CAM_FRONT'], seed=42)
    t1 = make_schedule(scene, scenario='temporal', **params)
    t2 = make_schedule(scene, scenario='temporal', **params)
    assert t1 == t2, 'Schedule must be fully deterministic'
    vals = [t1['samples'][f'{i:032x}']['CAM_FRONT']['rot_deg'] for i in range(9)]
    assert np.all(np.diff(vals) >= -1e-10) and abs(vals[-1]-10) < 1e-8
    assert t1['samples'][f'{8:032x}']['CAM_BACK']['rot_deg'] == 0
    shock = make_schedule(scene, scenario='shock', **params)
    assert shock['samples'][f'{3:032x}']['CAM_FRONT']['drift_alpha'] == 0
    assert shock['samples'][f'{4:032x}']['CAM_FRONT']['drift_alpha'] == 1
    Tgt = np.eye(4)
    Tgt[:3, 3] = [2, -1, 0.3]
    Dt = np.asarray(shock['samples'][f'{4:032x}']['CAM_FRONT']['delta_T'])
    Tb = Dt @ Tgt
    assert np.max(np.abs(np.linalg.inv(Dt) @ Tb - Tgt)) < 1e-10
    assert abs(np.linalg.det(Dt[:3,:3]) - 1) < 1e-10
    print('PASS: deterministic, temporal monotonicity, shock discontinuity, unaffected cams, SE(3) oracle')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--infos')
    p.add_argument('--nuscenes-root')
    p.add_argument('--nuscenes-version', default='v1.0-trainval')
    p.add_argument('--scenario', default='temporal', choices=['fixed','temporal','shock','shock_return'])
    p.add_argument('--rot-deg', type=float, default=2.0)
    p.add_argument('--trans-m', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--cameras', nargs='+', default=list(CAMERAS), choices=CAMERAS)
    p.add_argument('--shock-fraction', type=float, default=0.5)
    p.add_argument('--return-fraction', type=float, default=0.75)
    p.add_argument('--out')
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.infos or not args.out:
        p.error('--infos and --out are required without --self-test')
    frames = extract_scene_frames(args.infos, args.nuscenes_root, args.nuscenes_version)
    schedule = make_schedule(frames, scenario=args.scenario, rot_deg=args.rot_deg,
                             trans_m=args.trans_m, seed=args.seed, cameras=args.cameras,
                             shock_fraction=args.shock_fraction, return_fraction=args.return_fraction)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(schedule, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {out_path} :: scenario={args.scenario} '
          f'scenes={schedule["scene_count"]} samples={schedule["sample_count"]} '
          f'sha256={schedule["manifest_sha256"]}')


if __name__ == '__main__':
    main()
