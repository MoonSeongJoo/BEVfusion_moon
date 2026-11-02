#!/usr/bin/env python3
"""Evaluate LCCNet calibration + frozen Official BEVFusion detection.

Run from /workspace/mmdetection3d, using one GPU and the project's
OfficialBEVFusion implementation that supports calibration_mode='lccnet'.

Example:
    python -u tools/eval_lccnet_official_detail.py \
      --config projects/OfficialBEVFusion/configs/bevfusion_v110_lccnet.py \
      --detector-checkpoint /path/to/official_bevfusion.pth \
      --lccnet-checkpoint /path/to/best_joint_model.pth \
      --work-dir work_dirs/iv2027_lccnet_recheck/lccnet_official_full

Outputs:
    lccnet_full_detail.json: Overall/per-camera calibration and detection.
    lccnet_per_camera_summary.csv: Compact camera-wise summary.
    lccnet_sample_residuals.csv: Per-frame/per-camera residuals (dataset order).

Why a single GPU? The LCCNet implementation maintains Python-list calibration
accumulators inside each model process. Standard 2-GPU distributed inference
would leave those lists split by rank; this script therefore deliberately
runs non-distributed inference for exact aggregation.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

ACCUMULATOR_NAMES = {
    "broken_rot": "_lccnet_broken_rot_all",
    "corrected_rot": "_lccnet_pred_rot_all",
    "broken_trans": "_lccnet_broken_trans_all",
    "corrected_trans": "_lccnet_pred_trans_all",
}


def describe(values):
    """Summarize one full set of rotation (degrees) or translation (m)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("Calibration statistics are empty or non-finite")
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "count": int(x.size),
    }


def summarize_group(arrays):
    result = {name: describe(x) for name, x in arrays.items()}
    broken_rot = result["broken_rot"]["mean"]
    broken_trans = result["broken_trans"]["mean"]
    result["rotation_recovery_pct"] = (
        100.0 * (broken_rot - result["corrected_rot"]["mean"])
        / max(broken_rot, 1e-12)
    )
    result["translation_recovery_pct"] = (
        100.0 * (broken_trans - result["corrected_trans"]["mean"])
        / max(broken_trans, 1e-12)
    )
    return result


def collect_calibration(model):
    """Read the accumulators already collected by BEVFusion's LCCNet mode."""
    arrays = {}
    for name, attr in ACCUMULATOR_NAMES.items():
        if not hasattr(model, attr):
            raise RuntimeError(
                f"Missing model accumulator '{attr}'. "
                "Confirm that calibration_mode=lccnet was actually executed "
                "and that your OfficialBEVFusion code has O-5 running metrics."
            )
        arrays[name] = np.asarray(getattr(model, attr), dtype=np.float64)

    counts = {name: x.size for name, x in arrays.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Unequal accumulator counts: {counts}")

    count = next(iter(counts.values()))
    if count <= 0 or count % len(CAMERAS) != 0:
        raise RuntimeError(
            f"Expected nonempty, multiple-of-six camera records; got {count}"
        )

    frames = count // len(CAMERAS)
    by_frame = {
        name: x.reshape(frames, len(CAMERAS)) for name, x in arrays.items()
    }
    report = {
        "frames": frames,
        "camera_observations": count,
        "overall": summarize_group(arrays),
        "per_camera": {
            camera: summarize_group(
                {name: matrix[:, idx] for name, matrix in by_frame.items()}
            )
            for idx, camera in enumerate(CAMERAS)
        },
    }
    return report, by_frame


def write_per_camera_csv(output, report):
    columns = [
        "camera",
        "broken_rot_mean_deg", "corrected_rot_mean_deg", "rot_recovery_pct",
        "broken_rot_median_deg", "corrected_rot_median_deg",
        "broken_rot_p90_deg", "corrected_rot_p90_deg",
        "broken_trans_mean_m", "corrected_trans_mean_m", "trans_recovery_pct",
        "broken_trans_median_m", "corrected_trans_median_m",
        "broken_trans_p90_m", "corrected_trans_p90_m",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for cam, stats in report["per_camera"].items():
            writer.writerow({
                "camera": cam,
                "broken_rot_mean_deg": stats["broken_rot"]["mean"],
                "corrected_rot_mean_deg": stats["corrected_rot"]["mean"],
                "rot_recovery_pct": stats["rotation_recovery_pct"],
                "broken_rot_median_deg": stats["broken_rot"]["median"],
                "corrected_rot_median_deg": stats["corrected_rot"]["median"],
                "broken_rot_p90_deg": stats["broken_rot"]["p90"],
                "corrected_rot_p90_deg": stats["corrected_rot"]["p90"],
                "broken_trans_mean_m": stats["broken_trans"]["mean"],
                "corrected_trans_mean_m": stats["corrected_trans"]["mean"],
                "trans_recovery_pct": stats["translation_recovery_pct"],
                "broken_trans_median_m": stats["broken_trans"]["median"],
                "corrected_trans_median_m": stats["corrected_trans"]["median"],
                "broken_trans_p90_m": stats["broken_trans"]["p90"],
                "corrected_trans_p90_m": stats["corrected_trans"]["p90"],
            })


def write_raw_csv(output, by_frame):
    columns = ["frame_index", "camera", "broken_rot_deg", "corrected_rot_deg",
               "broken_trans_m", "corrected_trans_m"]
    nframes = by_frame["broken_rot"].shape[0]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for frame in range(nframes):
            for i, camera in enumerate(CAMERAS):
                writer.writerow([
                    frame, camera,
                    float(by_frame["broken_rot"][frame, i]),
                    float(by_frame["corrected_rot"][frame, i]),
                    float(by_frame["broken_trans"][frame, i]),
                    float(by_frame["corrected_trans"][frame, i]),
                ])


def json_safe(value):
    """Convert evaluator outputs to JSON types without losing metric values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args():
    p = argparse.ArgumentParser(
        description="Full LCCNet geometric-calibration and BEVFusion detection evaluation"
    )
    p.add_argument("--config", required=True, help="OfficialBEVFusion LCCNet config")
    p.add_argument("--detector-checkpoint", required=True, help="Official BEVFusion detector checkpoint")
    p.add_argument("--lccnet-checkpoint", required=True, help="Trained LCCNet checkpoint")
    p.add_argument("--work-dir", required=True, help="Output directory")
    p.add_argument("--max-rot-deg", type=float, default=10.0,
                   help="Axis-wise rotation perturbation bound in degrees")
    p.add_argument("--max-trans-m", type=float, default=0.75,
                   help="Axis-wise translation perturbation bound in meters")
    p.add_argument("--perturb-seed", type=int, default=20260811)
    p.add_argument("--expected-frames", type=int, default=0,
                   help="Fail if evaluation has a different frame count; 0 disables this check")
    p.add_argument("--skip-raw-csv", action="store_true",
                   help="Only save aggregate JSON and 6-camera summary CSV")
    return p.parse_args()


def main():
    args = parse_args()
    for label, file in (
        ("Config", args.config),
        ("Official detector checkpoint", args.detector_checkpoint),
        ("LCCNet checkpoint", args.lccnet_checkpoint),
    ):
        if not Path(file).is_file():
            raise FileNotFoundError(f"{label} not found: {file}")

    # Import training framework only after parsing CLI; --help works anywhere.
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet3d.utils import register_all_modules

    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(output_dir)
    cfg.load_from = str(Path(args.detector_checkpoint).resolve())
    cfg.resume = False
    cfg.launcher = "none"

    # No multi-rank test: model's calibration statistics live in Python lists.
    cfg.model.calibration_mode = "lccnet"
    cfg.model.lccnet_checkpoint = str(Path(args.lccnet_checkpoint).resolve())
    cfg.model.lccnet_use_feat_from = 1
    cfg.model.lccnet_depth_scale = 50.0
    cfg.model.lccnet_cam_chunk = 6
    cfg.model.perturb_max_rot_deg = args.max_rot_deg
    cfg.model.perturb_max_trans_m = args.max_trans_m
    cfg.model.perturb_seed = args.perturb_seed
    # Do not inherit a temporal/shock-drift manifest from an unrelated test.
    cfg.model.drift_manifest = None
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.sampler.shuffle = False

    print("\n[LCCNET + OFFICIAL BEVFUSION EVALUATION SETTINGS]", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Detector: {cfg.load_from}", flush=True)
    print(f"LCCNet: {cfg.model.lccnet_checkpoint}", flush=True)
    print(f"Perturbation: +/-{args.max_rot_deg} deg, +/-{args.max_trans_m} m per axis; seed={args.perturb_seed}", flush=True)

    runner = Runner.from_cfg(cfg)
    detection = runner.test()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    calibration, raw = collect_calibration(model)

    if args.expected_frames and calibration["frames"] != args.expected_frames:
        raise RuntimeError(
            f"Expected {args.expected_frames} frames but got {calibration['frames']}"
        )

    result = {
        "config": args.config,
        "detector_checkpoint": cfg.load_from,
        "lccnet_checkpoint": cfg.model.lccnet_checkpoint,
        "perturbation": {
            "rotation_bound_deg_per_axis": args.max_rot_deg,
            "translation_bound_m_per_axis": args.max_trans_m,
            "seed": args.perturb_seed,
            "convention": "The OfficialBEVFusion model determines the exact injected delta",
        },
        "calibration": calibration,
        "detection": json_safe(detection),
    }

    json_path = output_dir / "lccnet_full_detail.json"
    per_camera_path = output_dir / "lccnet_per_camera_summary.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_per_camera_csv(per_camera_path, calibration)
    if not args.skip_raw_csv:
        write_raw_csv(output_dir / "lccnet_sample_residuals.csv", raw)

    print("\n==================================================")
    print("[LCCNET FULL CALIBRATION + DETECTION REPORT]")
    print(f"Frames: {calibration['frames']}")
    print(f"Camera observations: {calibration['camera_observations']}")
    print("\n[OVERALL CALIBRATION]")
    print(json.dumps(calibration["overall"], indent=2, ensure_ascii=False))
    print("\n[PER-CAMERA CALIBRATION]")
    for camera, summary in calibration["per_camera"].items():
        print(camera, json.dumps(summary, ensure_ascii=False))
    print("\n[DETECTION METRICS]")
    print(json.dumps(json_safe(detection), ensure_ascii=False, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {per_camera_path}")
    if not args.skip_raw_csv:
        print(f"Saved: {output_dir / 'lccnet_sample_residuals.csv'}")
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()