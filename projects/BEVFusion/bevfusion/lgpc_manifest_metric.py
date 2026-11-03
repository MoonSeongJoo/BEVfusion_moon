import json
import os

import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

from projects.BEVFusion.bevfusion.calib_head import (
    axis_angle_to_matrix
)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    result = np.asarray(value, dtype=np.float64)

    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite calibration value")

    return result


@METRICS.register_module()
class LGPCManifestMetric(BaseMetric):

    default_prefix = "lgpc_export"

    def __init__(
        self,
        output_path="work_dirs/lgpc_official/lgpc_manifest.json",
        collect_device="cpu",
        prefix=None,
    ):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
        )

        self.output_path = output_path

    def process(self, data_batch, data_samples):

        for sample in data_samples:

            # ============================================
            # FIX: MMEngine Evaluator converts DataSample
            #      into dict before calling Metric.process()
            # ============================================

            if isinstance(sample, dict):

                # Some versions expose metainfo as a
                # nested dictionary, while others flatten
                # metadata into top-level dictionary keys.

                nested_meta = sample.get("metainfo")

                if isinstance(nested_meta, dict):
                    meta = {
                        **sample,
                        **nested_meta,
                    }
                else:
                    meta = sample

            elif hasattr(sample, "metainfo"):

                # Direct Det3DDataSample input
                meta = sample.metainfo

            else:

                raise TypeError(
                    f"Unsupported sample type: {type(sample)}"
                )

            # ============================================
            # Sample ID
            # ============================================

            sample_idx = meta.get("sample_idx")

            if sample_idx is None:
                sample_idx = meta.get("token")

            if sample_idx is None:
                raise RuntimeError(
                    "Missing sample_idx/token. "
                    f"Available keys: {list(meta.keys())}"
                )

            sample_idx = str(sample_idx)

            # ============================================
            # Verify all required calibration fields
            # ============================================

            required_keys = [
                "camera2lidar",
                "broken_camera2lidar",
                "broken_camera_intrinsics",
                "stage1_pred_delta_rot",
                "stage1_pred_delta_trans",
                "stage1_active_cam_indices",
            ]

            missing_keys = [
                key for key in required_keys
                if key not in meta
            ]

            if missing_keys:
                raise RuntimeError(
                    f"[LGPC EXPORT] Missing keys: {missing_keys}\n"
                    f"Available keys: {list(meta.keys())}"
                )

            # First-sample diagnostic
            if len(self.results) == 0:

                print("\n========================================")
                print("[LGPC EXPORT] METRIC INPUT CHECK")

                print("[LGPC EXPORT] Sample Type:", type(sample))

                print("[LGPC EXPORT] Sample ID:", sample_idx)

                print("[LGPC EXPORT] Required Fields: PASS")

                for key in required_keys:
                    value = meta[key]
                    shape = getattr(value, "shape", None)

                    print(
                        f"  {key}: shape={shape}, type={type(value).__name__}"
                    )

                print("========================================\n")

            if not sample_idx:
                raise RuntimeError("Missing sample_idx")

            # Ground-truth calibration
            # Used only for verification.
            clean = to_numpy(
                meta["camera2lidar"]
            )

            # Actual perturbed calibration
            broken = to_numpy(
                meta["broken_camera2lidar"]
            )

            intrinsics = to_numpy(
                meta["broken_camera_intrinsics"]
            )

            # LGPC Stage-1 predictions
            rot = to_numpy(
                meta["stage1_pred_delta_rot"]
            )

            trans = to_numpy(
                meta["stage1_pred_delta_trans"]
            )

            active_indices = [
                int(i)
                for i in meta["stage1_active_cam_indices"]
            ]

            if clean.shape != (6, 4, 4):
                raise RuntimeError(
                    f"Invalid clean shape: {clean.shape}"
                )

            if broken.shape != (6, 4, 4):
                raise RuntimeError(
                    f"Invalid broken shape: {broken.shape}"
                )

            if intrinsics.shape != (6, 4, 4):
                raise RuntimeError(
                    f"Invalid intrinsic shape: {intrinsics.shape}"
                )

            if rot.shape != (6, 3):
                raise RuntimeError(
                    f"Invalid rotation shape: {rot.shape}"
                )

            if trans.shape != (6, 3):
                raise RuntimeError(
                    f"Invalid translation shape: {trans.shape}"
                )

            if len(set(active_indices)) != len(active_indices):
                raise RuntimeError("Duplicate active camera index")

            if any(i < 0 or i >= 6 for i in active_indices):
                raise RuntimeError("Invalid active camera index")

            # ==========================================
            # LGPC Rotation: axis-angle -> 3x3
            # ==========================================

            rot_tensor = torch.as_tensor(
                rot,
                dtype=torch.float64,
            )

            rot_matrix = (
                axis_angle_to_matrix(rot_tensor)
                .detach()
                .cpu()
                .numpy()
            )

            # ==========================================
            # Build predicted 4x4 Delta matrix
            # ==========================================

            delta = np.tile(
                np.eye(4, dtype=np.float64),
                (6, 1, 1),
            )

            delta[:, :3, :3] = rot_matrix
            delta[:, :3, 3] = trans

            # ==========================================
            # Apply inverse correction
            #
            # T_corrected = inv(Delta) @ T_broken
            # ==========================================

            corrected = (
                np.linalg.inv(delta) @ broken
            )

            if not np.isfinite(corrected).all():
                raise RuntimeError(
                    "Invalid corrected calibration"
                )

            self.results.append({

                "sample_idx": sample_idx,

                "clean_cam2lidar": clean.tolist(),

                "broken_cam2lidar": broken.tolist(),

                "corrected_cam2lidar": corrected.tolist(),

                "cam2img": intrinsics.tolist(),

                "active_cam_indices": active_indices,

                "pred_delta_rot": rot.tolist(),

                "pred_delta_trans": trans.tolist(),

            })

    def compute_metrics(self, results):

        records = {}

        for record in results:

            sample_idx = record["sample_idx"]

            if sample_idx in records:
                raise RuntimeError(
                    f"Duplicate sample: {sample_idx}"
                )

            records[sample_idx] = record

        if len(records) == 0:
            raise RuntimeError("Empty LGPC export")

        output_path = os.path.abspath(
            self.output_path
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        manifest = {
            "version": 1,
            "num_samples": len(records),
            "records": records,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(manifest, f)

        inactive_count = sum(
            6 - len(r["active_cam_indices"])
            for r in records.values()
        )

        print(
            "[LGPC EXPORT] "
            f"frames={len(records)}, "
            f"inactive_cameras={inactive_count}"
        )

        print(
            "[LGPC EXPORT] Saved:",
            output_path,
        )

        return {
            "frames": float(len(records)),
            "inactive_cameras": float(inactive_count),
        }