# modify from https://github.com/mit-han-lab/bevfusion
import copy
from typing import Any, Dict, Optional, Tuple,List
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr
from .imageprocessing_unit import visualize_full_pipeline_enhanced,project_points_to_image,visualize_calibration_effect
from .calib_head import axis_angle_to_matrix,geodesic_distance_loss,correct_camera_proposals,quaternion_to_matrix,identity_matrix_loss
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

from collections import defaultdict
from mmengine.logging import MMLogger
from mmengine.dist import is_main_process

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

def _to_tensor(x: Any, device=None) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    try:
        return torch.as_tensor(x, device=device)
    except Exception:
        return None

def _squeeze_bn(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    # (1,N,3) -> (N,3)
    if x.dim() == 3 and x.shape[0] == 1:
        return x[0]
    return x


def _mean_any_shape_to_3(t: torch.Tensor) -> torch.Tensor:
    """Whatever shape (..., 3) -> (3,) by averaging all leading dims."""
    if t.numel() == 3 and t.shape == (3,):
        return t
    t = t.reshape(-1, 3)
    return t.mean(dim=0)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """aa: (...,3) axis-angle -> (...,3,3)"""
    orig_shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3).float()

    theta = torch.linalg.norm(aa, dim=1, keepdim=True).clamp(min=1e-12)
    k = aa / theta

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = torch.zeros_like(kx)

    K = torch.stack([
        zero, -kz,  ky,
        kz,  zero, -kx,
        -ky, kx,  zero
    ], dim=1).reshape(-1, 3, 3)

    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0).expand_as(K)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R.reshape(*orig_shape, 3, 3)


def geodesic_rot_error_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Ra,Rb: (...,3,3) -> (...,) degrees"""
    R = Ra.transpose(-1, -2) @ Rb
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    return ang * (180.0 / np.pi)

def _get_metainfo(x: Any) -> dict:
    """Det3DDataSample / InstanceData / dict 모두에서 stage1/2 키를 찾기 위해 flatten."""
    out = {}
    if x is None:
        return out

    # 1) metainfo (Det3DDataSample / BaseDataElement)
    if hasattr(x, 'metainfo'):
        try:
            out.update(dict(x.metainfo))
        except Exception:
            pass

    # 2) data fields (InstanceData는 repr이 dict처럼 보임)
    if isinstance(x, Mapping):
        out.update(dict(x))
    elif hasattr(x, 'items'):
        try:
            out.update(dict(x.items()))
        except Exception:
            pass

    # 3) nested pred_instances_3d / pred_instances inside a sample OR dict
    for key in ('pred_instances_3d', 'pred_instances'):
        y = None
        if hasattr(x, key):
            y = getattr(x, key)
        elif isinstance(x, Mapping) and key in x:
            y = x.get(key)

        if y is None:
            continue

        if hasattr(y, 'metainfo'):
            try:
                out.update(dict(y.metainfo))
            except Exception:
                pass

        if isinstance(y, Mapping):
            out.update(dict(y))
        elif hasattr(y, 'items'):
            try:
                out.update(dict(y.items()))
            except Exception:
                pass

    return out

def _pick(meta: dict, *keys):
    """meta에서 keys 순서대로 'None이 아닌 첫 값'을 반환 (Tensor도 안전)"""
    for k in keys:
        if k in meta:
            v = meta.get(k, None)
            if v is not None:
                return v
    return None

@METRICS.register_module()
class CalibRecoveryMetric(BaseMetric):
    def __init__(self, collect_device='cpu', prefix='calib_recovery', debug=False, debug_n=3):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.debug = debug
        self.debug_n = debug_n
        self._dbg_printed = 0
        self._skip = defaultdict(int) 

    def _get_gt_from_batch(self, gt_sample) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        """GT는 data_batch 쪽에서 읽는다."""
        meta = getattr(gt_sample, 'metainfo', {}) or {}

        gt_rot = meta.get('gt_delta_rot', None)
        gt_trans = meta.get('gt_delta_trans', None)

        # 혹시 attribute로 들어오는 구현이면 fallback
        if gt_rot is None and hasattr(gt_sample, 'gt_delta_rot'):
            gt_rot = getattr(gt_sample, 'gt_delta_rot', None)
        if gt_trans is None and hasattr(gt_sample, 'gt_delta_trans'):
            gt_trans = getattr(gt_sample, 'gt_delta_trans', None)

        gt_rot_t = _to_tensor(gt_rot)
        gt_trans_t = _to_tensor(gt_trans)
        if gt_rot_t is not None:
            gt_rot_t = _mean_any_shape_to_3(gt_rot_t)
        if gt_trans_t is not None:
            gt_trans_t = _mean_any_shape_to_3(gt_trans_t)
        return gt_rot_t, gt_trans_t

    def _get_stage1_from_pred(self, pred_sample) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        meta = _get_metainfo(pred_sample)

        pred1_rot = _pick(meta,
            'pred_delta_rot_1st',
            'stage1_pred_delta_rot',
            'stage1_pred_delta_rot_mean',
        )
        pred1_trans = _pick(meta,
            'pred_delta_trans_1st',
            'stage1_pred_delta_trans',
            'stage1_pred_delta_trans_mean',
        )

        r = _to_tensor(pred1_rot)
        t = _to_tensor(pred1_trans)
        if r is not None:
            r = _mean_any_shape_to_3(r)
        if t is not None:
            t = _mean_any_shape_to_3(t)
        return r, t

    def _get_stage2_from_pred(self, pred_sample) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        meta = _get_metainfo(pred_sample)

        pred2_rot = _pick(meta,
            'pred_delta_rot_2nd',
            'stage2_pred_delta_rot',
            'pred_delta_rot',          # 너가 기존에 stage2를 이 키로도 저장했었음
        )
        pred2_trans = _pick(meta,
            'pred_delta_trans_2nd',
            'stage2_pred_delta_trans',
            'pred_delta_trans',
        )

        r = _to_tensor(pred2_rot)
        t = _to_tensor(pred2_trans)
        if r is not None:
            r = _mean_any_shape_to_3(r)
        if t is not None:
            t = _mean_any_shape_to_3(t)
        return r, t

    def process(self, data_batch, data_samples):
        logger = MMLogger.get_current_instance()

        # GT는 data_batch에서, pred는 data_samples에서 꺼낸다
        gt_list = None
        if isinstance(data_batch, dict) and 'data_samples' in data_batch:
            gt_list = data_batch['data_samples']

        if gt_list is None:
            # 이 경우는 dataloader가 GT를 안 주는 구조 (test set / format_only 등)
            self._skip['no_data_batch_gt_list'] += len(data_samples)
            return

        for gt_s, pred_s in zip(gt_list, data_samples):
            self._skip['total'] += 1

            gt_rot_t, gt_trans_t = self._get_gt_from_batch(gt_s)
            if gt_rot_t is None or gt_trans_t is None:
                self._skip['no_gt'] += 1
                continue

            pred1_rot_t, pred1_trans_t = self._get_stage1_from_pred(pred_s)
            pred2_rot_t, pred2_trans_t = self._get_stage2_from_pred(pred_s)

            has_s1 = (pred1_rot_t is not None) and (pred1_trans_t is not None)
            has_s2 = (pred2_rot_t is not None) and (pred2_trans_t is not None)

            if not has_s1:
                self._skip['no_stage1'] += 1
                pred1_rot_t = torch.zeros_like(gt_rot_t)
                pred1_trans_t = torch.zeros_like(gt_trans_t)

            if not has_s2:
                self._skip['no_stage2'] += 1
                pred2_rot_t = torch.zeros_like(gt_rot_t)
                pred2_trans_t = torch.zeros_like(gt_trans_t)

            # rotations
            R_gt = axis_angle_to_matrix(gt_rot_t[None])          # (1,3,3)
            R1   = axis_angle_to_matrix(pred1_rot_t[None])       # (1,3,3)
            R2   = axis_angle_to_matrix(pred2_rot_t[None])       # (1,3,3)
            R0   = torch.eye(3, device=R_gt.device, dtype=R_gt.dtype)[None]  # (1,3,3)

            # ✅ residual target for stage2 (s2가 맞춰야 하는 남은 오차)
            R_res = R_gt @ R1.transpose(-1, -2)   # (1,3,3)

            # final compose (s2@ s1)
            R_total = R2 @ R1

            # translations
            t0 = torch.zeros_like(gt_trans_t)
            t1 = pred1_trans_t
            t2 = pred2_trans_t
            t_total = t1 + t2

            # ✅ residual target for stage2 translation
            t_res = gt_trans_t - t1

            # ---- metrics ----
            rot_before = geodesic_rot_error_deg(R0, R_gt)[0].item()
            rot_s1     = geodesic_rot_error_deg(R1, R_gt)[0].item()

            # ✅ s2 "단독" 평가는 R2 vs R_res (잔여 오차를 얼마나 잘 맞추는지)
            rot_s2     = geodesic_rot_error_deg(R2, R_res)[0].item() if has_s2 else float('nan')

            rot_final  = geodesic_rot_error_deg(R_total, R_gt)[0].item()

            trans_before = torch.norm(t0 - gt_trans_t, p=2).item()
            trans_s1     = torch.norm(t1 - gt_trans_t, p=2).item()
            trans_s2     = torch.norm(t2 - t_res, p=2).item() if has_s2 else float('nan')
            trans_final  = torch.norm(t_total - gt_trans_t, p=2).item()

            self.results.append(dict(
                rot_before_deg=rot_before,
                rot_s1_deg=rot_s1,
                rot_s2_deg=rot_s2,          # ✅ 추가
                rot_final_deg=rot_final,
                trans_before_m=trans_before,
                trans_s1_m=trans_s1,
                trans_s2_m=trans_s2,        # ✅ 추가
                trans_final_m=trans_final,
                has_s2=float(has_s2),       # ✅ NaN 평균낼 때 유용 (옵션)
            ))
            self._skip['appended'] += 1

            if self.debug and is_main_process() and self._dbg_printed < self.debug_n:
                pm = _get_metainfo(pred_s)
                gm = _get_metainfo(gt_s)
                s1_ok = ('pred_delta_rot_1st' in pm) or ('stage1_pred_delta_rot' in pm)
                s2_ok = (
                            ('pred_delta_rot_2nd' in pm) or
                            ('stage2_pred_delta_rot' in pm) or
                            ('pred_delta_rot' in pm)   # ✅ 너의 기존 stage2 저장 키까지 포함
                    )

                logger.info(
                    f"[CalibRecoveryMetric] sample_idx={gm.get('sample_idx', None)} "
                    f"GT(meta) keys has rot/trans={('gt_delta_rot' in gm)}/{('gt_delta_trans' in gm)} | "
                    f"PRED(meta) has s1={s1_ok}, "
                    f"s2={s2_ok}"
                )
                self._dbg_printed += 1

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        if is_main_process():
            logger.info(f"[CalibRecoveryMetric] skip_stats={dict(self._skip)}")

        if len(results) == 0:
            return dict()

        keys = results[0].keys()
        out = {}
        for k in keys:
            vals = np.array([r.get(k, np.nan) for r in results], dtype=np.float64)
            out[k] = float(np.nanmean(vals))
        return out

# ============================================================
# LCCNet-compatible physical calibration metric
# ============================================================

NUSC_CAMERA_NAMES_METRIC = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def _to_cam3_no_mean(x, device=None):
    """
    Convert calibration parameter to [Ncam, 3].

    IMPORTANT:
    Unlike old CalibRecoveryMetric,
    NEVER average cameras here.
    """

    if x is None:
        return None

    if torch.is_tensor(x):
        t = x.to(device) if device is not None else x
    else:
        try:
            t = torch.as_tensor(
                x,
                device=device,
                dtype=torch.float32
            )
        except Exception:
            return None

    t = t.float()

    # [1, Ncam, 3] -> [Ncam, 3]
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]

    # [3] -> [1,3]
    if t.dim() == 1:
        if t.numel() != 3:
            return None
        t = t.reshape(1, 3)

    if t.shape[-1] != 3:
        return None

    return t.reshape(-1, 3)


def _to_cam44(x, device=None):
    """
    Convert extrinsic matrices to [Ncam, 4, 4].

    Supported:
        [4,4]
        [Ncam,4,4]
        [1,Ncam,4,4]
        [Ncam,3,4]
    """

    if x is None:
        return None

    if torch.is_tensor(x):
        t = x.to(device) if device is not None else x
    else:
        try:
            t = torch.as_tensor(
                x,
                device=device,
                dtype=torch.float32
            )
        except Exception:
            return None

    t = t.float()

    # [1,N,4,4] -> [N,4,4]
    if t.dim() == 4 and t.shape[0] == 1:
        t = t[0]

    # [4,4] -> [1,4,4]
    if t.dim() == 2:
        t = t.unsqueeze(0)

    # [N,3,4] -> [N,4,4]
    if t.dim() == 3 and t.shape[-2:] == (3, 4):

        n = t.shape[0]

        bottom = torch.zeros(
            n, 1, 4,
            dtype=t.dtype,
            device=t.device
        )

        bottom[:, 0, 3] = 1.0

        t = torch.cat(
            [t, bottom],
            dim=1
        )

    if (
        t.dim() != 3
        or t.shape[-2:] != (4, 4)
    ):
        return None

    return t

def _build_delta_matrix(rot_aa, trans):
    """
    PCC prediction:
        rot_aa : [N,3] axis-angle
        trans  : [N,3]

    Returns:
        Delta_pred : [N,4,4]
    """

    R = axis_angle_to_matrix(
        rot_aa
    )

    n = R.shape[0]

    T = torch.eye(
        4,
        dtype=R.dtype,
        device=R.device
    ).unsqueeze(0).repeat(
        n, 1, 1
    )

    T[:, :3, :3] = R
    T[:, :3, 3] = trans

    return T

def _compute_lccnet_physical_residual(
    correction,
    broken_c2l,
    gt_c2l,
):
    """
    EXACT same physical residual definition
    used by current LCCNet validation.

    corrected_c2l =
        correction @ broken_c2l

    residual =
        corrected_c2l @ inv(gt_c2l)

    Perfect correction:
        residual == Identity

    Returns:
        rot_deg : [N]
        trans_m : [N]
    """

    corrected_c2l = (
        correction
        @ broken_c2l
    )

    residual = (
        corrected_c2l
        @ torch.linalg.inv(
            gt_c2l
        )
    )

    # --------------------------------------------------------
    # Translation residual
    # --------------------------------------------------------
    trans_m = torch.linalg.norm(
        residual[..., :3, 3],
        dim=-1
    )

    # --------------------------------------------------------
    # Rotation geodesic residual
    # Same implementation as LCCNet validation.
    # --------------------------------------------------------
    R = residual[..., :3, :3]

    trace = (
        R[..., 0, 0]
        + R[..., 1, 1]
        + R[..., 2, 2]
    )

    cosine = (
        trace - 1.0
    ) / 2.0

    cosine = torch.clamp(
        cosine,
        -1.0,
        1.0
    )

    rot_deg = torch.rad2deg(
        torch.acos(
            cosine
        )
    )

    return rot_deg, trans_m

def _extract_camera_names_metric(
    gt_sample,
    num_cams,
):
    meta = _get_metainfo(
        gt_sample
    )

    img_paths = meta.get(
        'img_path',
        None
    )

    if isinstance(
        img_paths,
        str
    ):
        img_paths = [img_paths]

    if not isinstance(
        img_paths,
        (list, tuple)
    ):
        return [
            f'CAMERA_INDEX_{i}'
            for i in range(num_cams)
        ]

    output = []

    for i in range(num_cams):

        if i >= len(img_paths):
            output.append(
                f'CAMERA_INDEX_{i}'
            )
            continue

        text = str(
            img_paths[i]
        ).replace(
            '\\',
            '/'
        )

        detected = None

        for name in (
            NUSC_CAMERA_NAMES_METRIC
        ):

            if f'/{name}/' in text:
                detected = name
                break

        if detected is None:
            detected = (
                f'CAMERA_INDEX_{i}'
            )

        output.append(
            detected
        )

    return output

@METRICS.register_module()
class CalibLCCNetMetric(BaseMetric):
    """
    PCC calibration metric using EXACTLY the same
    physical residual definition as the current
    LCCNet validation.

    Primary use:
        Apples-to-apples PCC vs LCCNet comparison.

    PCC convention
    --------------

        T_broken =
            Delta_GT @ T_GT

    Stage1 LGPC predicts:
        Delta1_pred

    Therefore correction:
        C1 =
            inv(Delta1_pred)

    Corrected calibration:
        T_corr_s1 =
            C1 @ T_broken

    LCCNet-compatible residual:
        E =
            T_corr_s1 @ inv(T_GT)

    Metrics:
        rotation = SO(3) geodesic angle [deg]
        translation = norm(E[:3,3]) [m]

    Camera handling:
        ERROR FIRST -> THEN aggregate.

    Never:
        mean camera parameters -> error.
    """

    def __init__(
        self,
        collect_device='cpu',
        prefix='calib_lccnet',
        debug=False,
        debug_n=3,
    ):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix
        )

        self.debug = debug
        self.debug_n = debug_n
        self._dbg_printed = 0

    # ========================================================
    # Geometry
    # ========================================================
    def _get_geometry(
        self,
        gt_sample,
    ):

        meta = _get_metainfo(
            gt_sample
        )

        gt_c2l = _pick(
            meta,
            'camera2lidar',
            'clean_camera2lidar',
            'cam2lidar_clean',
        )

        broken_c2l = _pick(
            meta,
            'broken_camera2lidar',
            'cam2lidar_broken',
        )

        gt_c2l = _to_cam44(
            gt_c2l
        )

        broken_c2l = _to_cam44(
            broken_c2l
        )

        return (
            gt_c2l,
            broken_c2l
        )

    # ========================================================
    # Stage1 LGPC
    # ========================================================
    def _get_stage1(
        self,
        pred_sample,
    ):

        meta = _get_metainfo(
            pred_sample
        )

        rot = _pick(
            meta,
            'pred_delta_rot_1st',
            'stage1_pred_delta_rot',
            'stage1_pred_delta_rot_mean',
        )

        trans = _pick(
            meta,
            'pred_delta_trans_1st',
            'stage1_pred_delta_trans',
            'stage1_pred_delta_trans_mean',
        )

        rot = _to_cam3_no_mean(
            rot
        )

        trans = _to_cam3_no_mean(
            trans
        )

        return rot, trans

    # ========================================================
    # Optional Stage2 residual SE(3)
    # ========================================================
    def _get_stage2(
        self,
        pred_sample,
    ):

        meta = _get_metainfo(
            pred_sample
        )

        rot = _pick(
            meta,
            'pred_delta_rot_2nd',
            'stage2_pred_delta_rot',
            'pred_delta_rot',
        )

        trans = _pick(
            meta,
            'pred_delta_trans_2nd',
            'stage2_pred_delta_trans',
            'pred_delta_trans',
        )

        rot = _to_cam3_no_mean(
            rot
        )

        trans = _to_cam3_no_mean(
            trans
        )

        return rot, trans

    # ========================================================
    # Broadcast Stage2 global prediction if necessary
    # ========================================================
    def _match_camera_count(
        self,
        x,
        ncam,
    ):

        if x is None:
            return None

        # Stage2 current RRRF:
        # [1,3] global scene-level prediction.
        if (
            x.shape[0] == 1
            and ncam > 1
        ):
            x = x.repeat(
                ncam,
                1
            )

        if x.shape[0] != ncam:
            return None

        return x

    # ========================================================
    # Process
    # ========================================================
    def process(
        self,
        data_batch,
        data_samples,
    ):

        logger = (
            MMLogger.get_current_instance()
        )

        if (
            not isinstance(
                data_batch,
                dict
            )
            or 'data_samples'
            not in data_batch
        ):
            return

        gt_list = data_batch[
            'data_samples'
        ]

        for gt_s, pred_s in zip(
            gt_list,
            data_samples
        ):
            
            # ========================================================
            # NEW:
            # Which calibration was ACTUALLY applied by BEVFusion?
            # ========================================================

            pred_meta = _get_metainfo(
                pred_s
            )

            gt_meta = _get_metainfo(
                gt_s
            )

            calibration_mode = pred_meta.get(
                'calibration_mode',
                None
            )

            if calibration_mode is None:
                calibration_mode = gt_meta.get(
                    'calibration_mode',
                    None
                )

            # 기존 checkpoint / 예전 test와 호환
            if calibration_mode is None:
                calibration_mode = 'pcc_full'

            # ------------------------------------------------
            # GT + Broken C2L
            # ------------------------------------------------
            (
                gt_c2l,
                broken_c2l
            ) = self._get_geometry(
                gt_s
            )

            if (
                gt_c2l is None
                or broken_c2l is None
            ):
                continue

            if (
                gt_c2l.shape[0]
                != broken_c2l.shape[0]
            ):
                continue

            ncam = gt_c2l.shape[0]

            # ------------------------------------------------
            # Stage1 prediction
            # ------------------------------------------------
            rot1, trans1 = (
                self._get_stage1(
                    pred_s
                )
            )

            rot1 = (
                self._match_camera_count(
                    rot1,
                    ncam
                )
            )

            trans1 = (
                self._match_camera_count(
                    trans1,
                    ncam
                )
            )

            if (
                rot1 is None
                or trans1 is None
            ):
                continue

            device = rot1.device

            gt_c2l = gt_c2l.to(
                device
            )

            broken_c2l = (
                broken_c2l.to(
                    device
                )
            )

            trans1 = trans1.to(
                device
            )

            # ------------------------------------------------
            # BROKEN metric
            #
            # LCCNet uses identity correction here.
            # ------------------------------------------------
            identity = torch.eye(
                4,
                dtype=gt_c2l.dtype,
                device=device
            ).unsqueeze(0).repeat(
                ncam,
                1,
                1
            )

            (
                broken_rot,
                broken_trans
            ) = (
                _compute_lccnet_physical_residual(
                    identity,
                    broken_c2l,
                    gt_c2l,
                )
            )

            # ------------------------------------------------
            # PCC Stage1:
            #
            # PCC predicts ERROR Delta1,
            # unlike LCCNet which predicts correction.
            #
            # Therefore:
            #
            #     C1 = inv(Delta1_pred)
            # ------------------------------------------------
            Delta1_pred = (
                _build_delta_matrix(
                    rot1,
                    trans1
                )
            )

            correction1 = (
                torch.linalg.inv(
                    Delta1_pred
                )
            )

            (
                s1_rot,
                s1_trans
            ) = (
                _compute_lccnet_physical_residual(
                    correction1,
                    broken_c2l,
                    gt_c2l,
                )
            )

            # ------------------------------------------------
            # Optional Stage2
            # ------------------------------------------------
            rot2, trans2 = (
                self._get_stage2(
                    pred_s
                )
            )

            has_s2 = (
                rot2 is not None
                and trans2 is not None
            )

            if has_s2:

                rot2 = (
                    self._match_camera_count(
                        rot2,
                        ncam
                    )
                )

                trans2 = (
                    self._match_camera_count(
                        trans2,
                        ncam
                    )
                )

                has_s2 = (
                    rot2 is not None
                    and trans2 is not None
                )

            if has_s2:

                rot2 = rot2.to(
                    device
                )

                trans2 = trans2.to(
                    device
                )

                Delta2_pred = (
                    _build_delta_matrix(
                        rot2,
                        trans2
                    )
                )

                correction2 = (
                    torch.linalg.inv(
                        Delta2_pred
                    )
                )

            #     # --------------------------------------------
            #     # Exact geometric composition:
            #     #
            #     # T_final =
            #     # C2 @ C1 @ T_broken
            #     #
            #     # We can pass C2@C1 directly into the
            #     # LCCNet residual helper.
            #     # --------------------------------------------
            #     correction_final = (
            #         correction2
            #         @ correction1
            #     )

            # else:

            #     # feature_refine / lgpc_only:
            #     # no additional physical calibration.
            #     correction_final = (
            #         correction1
            #     )

            # ============================================================
            # FINAL = correction ACTUALLY applied to downstream perception
            # ============================================================

            if calibration_mode == 'pcc_broken_refine':

                # --------------------------------------------------------
                # LGPC prediction exists,
                # but it was NOT applied to physical calibration.
                #
                # Therefore actual final extrinsic remains BROKEN.
                # --------------------------------------------------------
                correction_final = identity


            elif calibration_mode == 'broken':

                # Pure Broken BEVFusion baseline
                correction_final = identity


            elif calibration_mode in {
                'clean',
                'oracle',
                'pcc_clean_refine',
            }:

                # --------------------------------------------------------
                # corrected_c2l should be GT/clean.
                #
                # We need C such that:
                #
                # C @ T_broken = T_gt
                #
                # therefore:
                #
                # C = T_gt @ inv(T_broken)
                # --------------------------------------------------------
                correction_final = (
                    gt_c2l
                    @ torch.linalg.inv(
                        broken_c2l
                    )
                )


            elif calibration_mode in {
                'pcc_calib_only',
                'pcc_full',
            }:

                # PCC Stage1 physically applied

                if has_s2:

                    correction_final = (
                        correction2
                        @ correction1
                    )

                else:

                    correction_final = (
                        correction1
                    )


            else:

                # 기존 behavior fallback
                if has_s2:

                    correction_final = (
                        correction2
                        @ correction1
                    )

                else:

                    correction_final = (
                        correction1
                    )

            (
                final_rot,
                final_trans
            ) = (
                _compute_lccnet_physical_residual(
                    correction_final,
                    broken_c2l,
                    gt_c2l,
                )
            )

            # ------------------------------------------------
            # Camera names
            # ------------------------------------------------
            camera_names = (
                _extract_camera_names_metric(
                    gt_s,
                    ncam
                )
            )

            # ------------------------------------------------
            # Store EACH CAMERA independently
            # ------------------------------------------------
            for cam_idx in range(
                ncam
            ):

                self.results.append(
                    dict(
                        camera_name=(
                            camera_names[
                                cam_idx
                            ]
                        ),

                        has_s2=float(
                            has_s2
                        ),

                        broken_rot_deg=float(
                            broken_rot[
                                cam_idx
                            ].detach().cpu()
                        ),

                        broken_trans_m=float(
                            broken_trans[
                                cam_idx
                            ].detach().cpu()
                        ),

                        s1_rot_deg=float(
                            s1_rot[
                                cam_idx
                            ].detach().cpu()
                        ),

                        s1_trans_m=float(
                            s1_trans[
                                cam_idx
                            ].detach().cpu()
                        ),

                        final_rot_deg=float(
                            final_rot[
                                cam_idx
                            ].detach().cpu()
                        ),

                        final_trans_m=float(
                            final_trans[
                                cam_idx
                            ].detach().cpu()
                        ),
                    )
                )

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------
            if (
                self.debug
                and is_main_process()
                and self._dbg_printed
                < self.debug_n
            ):

                meta = _get_metainfo(
                    gt_s
                )

                logger.info(
                    "[CalibLCCNetMetric] "
                    f"sample={meta.get('sample_idx', None)} "
                    f"mode={calibration_mode} "
                    f"ncam={ncam} "
                    f"has_s2={has_s2} | "
                    f"BROKEN "
                    f"Rot={broken_rot.mean().item():.4f}deg "
                    f"Trans={broken_trans.mean().item():.4f}m | "
                    f"S1 "
                    f"Rot={s1_rot.mean().item():.4f}deg "
                    f"Trans={s1_trans.mean().item():.4f}m | "
                    f"FINAL "
                    f"Rot={final_rot.mean().item():.4f}deg "
                    f"Trans={final_trans.mean().item():.4f}m"
                )

                self._dbg_printed += 1

    # ========================================================
    # Aggregate
    # ========================================================
    def compute_metrics(
        self,
        results,
    ):

        logger = (
            MMLogger.get_current_instance()
        )

        if len(results) == 0:
            return {}

        # ----------------------------------------------------
        # Helper
        # ----------------------------------------------------
        def values(key, rows=results):

            return np.asarray(
                [
                    float(r[key])
                    for r in rows
                ],
                dtype=np.float64
            )

        def summary(x):

            return dict(
                mean=float(
                    np.mean(x)
                ),
                median=float(
                    np.median(x)
                ),
                p90=float(
                    np.quantile(
                        x,
                        0.90
                    )
                ),
                max=float(
                    np.max(x)
                ),
            )

        broken_rot = values(
            'broken_rot_deg'
        )

        broken_trans = values(
            'broken_trans_m'
        )

        s1_rot = values(
            's1_rot_deg'
        )

        s1_trans = values(
            's1_trans_m'
        )

        final_rot = values(
            'final_rot_deg'
        )

        final_trans = values(
            'final_trans_m'
        )

        br = summary(
            broken_rot
        )

        bt = summary(
            broken_trans
        )

        sr = summary(
            s1_rot
        )

        st = summary(
            s1_trans
        )

        fr = summary(
            final_rot
        )

        ft = summary(
            final_trans
        )

        # ----------------------------------------------------
        # EXACT same recovery definition as LCCNet.
        # ----------------------------------------------------
        s1_rot_recovery = (
            100.0
            * (
                1.0
                - sr['mean']
                / max(
                    br['mean'],
                    1e-12
                )
            )
        )

        s1_trans_recovery = (
            100.0
            * (
                1.0
                - st['mean']
                / max(
                    bt['mean'],
                    1e-12
                )
            )
        )

        final_rot_recovery = (
            100.0
            * (
                1.0
                - fr['mean']
                / max(
                    br['mean'],
                    1e-12
                )
            )
        )

        final_trans_recovery = (
            100.0
            * (
                1.0
                - ft['mean']
                / max(
                    bt['mean'],
                    1e-12
                )
            )
        )

        # Same LCCNet joint score:
        # smaller = better.
        s1_joint_score = (
            sr['mean']
            / max(
                br['mean'],
                1e-12
            )
            +
            st['mean']
            / max(
                bt['mean'],
                1e-12
            )
        )

        final_joint_score = (
            fr['mean']
            / max(
                br['mean'],
                1e-12
            )
            +
            ft['mean']
            / max(
                bt['mean'],
                1e-12
            )
        )

        # ====================================================
        # Per-camera summaries
        # ====================================================
        per_camera = {}

        camera_names = (
            NUSC_CAMERA_NAMES_METRIC
        )

        for cam_name in camera_names:

            rows = [
                r
                for r in results
                if r.get(
                    'camera_name',
                    ''
                ) == cam_name
            ]

            if len(rows) == 0:
                continue

            per_camera[
                cam_name
            ] = dict(
                broken_rot=float(
                    np.mean(
                        values(
                            'broken_rot_deg',
                            rows
                        )
                    )
                ),

                s1_rot=float(
                    np.mean(
                        values(
                            's1_rot_deg',
                            rows
                        )
                    )
                ),

                final_rot=float(
                    np.mean(
                        values(
                            'final_rot_deg',
                            rows
                        )
                    )
                ),

                broken_trans=float(
                    np.mean(
                        values(
                            'broken_trans_m',
                            rows
                        )
                    )
                ),

                s1_trans=float(
                    np.mean(
                        values(
                            's1_trans_m',
                            rows
                        )
                    )
                ),

                final_trans=float(
                    np.mean(
                        values(
                            'final_trans_m',
                            rows
                        )
                    )
                ),
            )

        # ====================================================
        # LCCNet-style console print
        # ====================================================
        if is_main_process():

            logger.info(
                "\n"
                "============================================================\n"
                "[PCC LCCNET-COMPATIBLE CALIBRATION METRIC]\n"
                "\n"
                "[BROKEN]\n"
                f"Rot   mean={br['mean']:.6f} deg   "
                f"median={br['median']:.6f} deg   "
                f"P90={br['p90']:.6f} deg\n"
                f"Trans mean={bt['mean']:.6f} m     "
                f"median={bt['median']:.6f} m     "
                f"P90={bt['p90']:.6f} m\n"
                "\n"
                "[PCC LGPC Stage1]\n"
                f"Rot   mean={sr['mean']:.6f} deg   "
                f"median={sr['median']:.6f} deg   "
                f"P90={sr['p90']:.6f} deg\n"
                f"Trans mean={st['mean']:.6f} m     "
                f"median={st['median']:.6f} m     "
                f"P90={st['p90']:.6f} m\n"
                "\n"
                f"Rotation recovery    = "
                f"{s1_rot_recovery:.2f}%\n"
                f"Translation recovery = "
                f"{s1_trans_recovery:.2f}%\n"
                f"Joint score          = "
                f"{s1_joint_score:.6f}\n"
                "\n"
                "[PCC FINAL]\n"
                f"Rot   mean={fr['mean']:.6f} deg   "
                f"median={fr['median']:.6f} deg   "
                f"P90={fr['p90']:.6f} deg\n"
                f"Trans mean={ft['mean']:.6f} m     "
                f"median={ft['median']:.6f} m     "
                f"P90={ft['p90']:.6f} m\n"
                "\n"
                f"Final rotation recovery    = "
                f"{final_rot_recovery:.2f}%\n"
                f"Final translation recovery = "
                f"{final_trans_recovery:.2f}%\n"
                f"Final joint score          = "
                f"{final_joint_score:.6f}\n"
                "\n"
                "[PER CAMERA]"
            )

            for (
                cam_name,
                x
            ) in per_camera.items():

                logger.info(
                    f"{cam_name:<16s} "
                    f"Rot "
                    f"{x['broken_rot']:.3f} "
                    f"-> {x['s1_rot']:.3f} "
                    f"-> {x['final_rot']:.3f} deg   "
                    f"Trans "
                    f"{x['broken_trans']:.3f} "
                    f"-> {x['s1_trans']:.3f} "
                    f"-> {x['final_trans']:.3f} m"
                )

            logger.info(
                "============================================================"
            )

        # ====================================================
        # MMEngine output
        # ====================================================
        return dict(
            broken_rot_mean_deg=br['mean'],
            broken_rot_median_deg=br['median'],
            broken_rot_p90_deg=br['p90'],

            broken_trans_mean_m=bt['mean'],
            broken_trans_median_m=bt['median'],
            broken_trans_p90_m=bt['p90'],

            s1_rot_mean_deg=sr['mean'],
            s1_rot_median_deg=sr['median'],
            s1_rot_p90_deg=sr['p90'],

            s1_trans_mean_m=st['mean'],
            s1_trans_median_m=st['median'],
            s1_trans_p90_m=st['p90'],

            s1_rotation_recovery_pct=(
                s1_rot_recovery
            ),

            s1_translation_recovery_pct=(
                s1_trans_recovery
            ),

            s1_joint_score=(
                s1_joint_score
            ),

            final_rot_mean_deg=fr['mean'],
            final_rot_median_deg=fr['median'],
            final_rot_p90_deg=fr['p90'],

            final_trans_mean_m=ft['mean'],
            final_trans_median_m=ft['median'],
            final_trans_p90_m=ft['p90'],

            final_rotation_recovery_pct=(
                final_rot_recovery
            ),

            final_translation_recovery_pct=(
                final_trans_recovery
            ),

            final_joint_score=(
                final_joint_score
            ),

            has_s2_ratio=float(
                np.mean(
                    values(
                        'has_s2'
                    )
                )
            ),
        )

@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class TransFusionHead(nn.Module):

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',

        loss_cls=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean'
        ),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean'
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean'
        ),

        # NEW
        rrrf_mode='residual_se3',
        query_source='fused',

        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.rrrf_mode = rrrf_mode

        assert self.rrrf_mode in [
            'residual_se3',
            'feature_refine',
            'lgpc_only'
        ], f'Unknown RRRF mode: {self.rrrf_mode}'

        print(f'[RRRF] mode = {self.rrrf_mode}')

        self.query_source = query_source

        assert self.query_source in [
            'fused',
            'lidar',
        ], (
            f'Unknown query_source: '
            f'{self.query_source}'
        )

        print(
            f'[RRRF] query_source = '
            f'{self.query_source}'
        )

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes  += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                self.num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))
            
        # 1단계 퓨전을 위한 새로운 전용 모듈들을 정의합니다.
        self.fusion_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channel,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True  # [Batch, Seq, Channel] 입력을 받도록 설정
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.ReLU(),
            nn.Linear(hidden_channel * 2, hidden_channel),
        )
        self.fusion_norm1 = nn.LayerNorm(hidden_channel)
        self.fusion_norm2 = nn.LayerNorm(hidden_channel)

        # 카메라 제안(det_xyz)의 3D 좌표를 위한 Positional Embedding (유지)
        self.camera_proposal_pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel)
        )

        # BEV 쿼리(query_pos)의 2D 좌표를 위한 Positional Embedding (신규 추가)
        # MMDetection 레이어 내부 기능을 밖으로 꺼내온 것입니다.
        self.bev_query_pos_embedding = nn.Sequential(
            nn.Linear(2, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel),
        )
        calibration_input_dim = hidden_channel
        calibration_hidden_dim = 256 # Intermediate dimension, can be tuned

        ########### old calibration_predictor version ##############
        self.calibration_predictor = nn.Sequential(
            nn.Linear(calibration_input_dim, calibration_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(calibration_hidden_dim, 6),  # (rot 3 + trans 3)
        )

        # Stage-2 SE(3) regressor is not trained in feature_refine/lgpc_only mode.
        if self.rrrf_mode != 'residual_se3':
            for param in self.calibration_predictor.parameters():
                param.requires_grad = False

        # # 🛑 1. 특징 추출기 (SHARED HEAD) - Input -> Hidden
        # self.calib_shared_fc = nn.Sequential(
        #     nn.Linear(calibration_input_dim, calibration_hidden_dim),
        #     nn.ReLU() # Or nn.LeakyReLU(0.01)
        # )

        # # 🛑 수정 후 (Capacity 및 Stability 증가):
        # # 🛑 2. 회전 예측 브랜치 (DECOUPLED ROTATION)
        # self.calib_rot_predictor = nn.Sequential(
        #     nn.Linear(calibration_hidden_dim, 512),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(32, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3) # 최종 Axis-Angle 출력
        # )

        # # 🛑 3. 이동 예측 브랜치 (DECOUPLED TRANSLATION)
        # self.calib_trans_predictor = nn.Sequential(
        #     nn.Linear(calibration_hidden_dim, 512),
        #     nn.GroupNorm(32, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(32, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 3) # 최종 Delta XYZ 출력
        # )

        # # --- ✨ 4. Zero Initialization 적용 위치 ---
        # # nn.Sequential 내부의 마지막 Linear 레이어에 적용해야 합니다.
        # torch.nn.init.constant_(self.calib_rot_predictor[-1].weight.data, 0.)
        # torch.nn.init.constant_(self.calib_rot_predictor[-1].bias.data, 0.)

        # torch.nn.init.constant_(self.calib_trans_predictor[-1].weight.data, 0.)
        # torch.nn.init.constant_(self.calib_trans_predictor[-1].bias.data, 0.)



        # --- ✨ 추가: 2단계 정제 퓨전을 위한 레이어들 ✨ ---
        self.refined_attention = nn.MultiheadAttention(
            embed_dim=hidden_channel,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.refined_ffn = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.ReLU(), # 또는 LeakyReLU
            nn.Linear(hidden_channel * 2, hidden_channel),
        )
        self.refined_norm1 = nn.LayerNorm(hidden_channel)
        self.refined_norm2 = nn.LayerNorm(hidden_channel)

        # --- ✨ 추가: 2단계 퓨전 결과를 평가하기 위한 보조 헤드 ✨ ---
        # 기존 prediction_heads의 마지막 레이어와 유사한 구조 사용
        aux_heads = copy.deepcopy(common_heads)
        aux_heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
        self.refined_fusion_aux_head = SeparateHead(
            hidden_channel,
            aux_heads,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
        )
        # ---------------------------------------------------

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.training_step = 0

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x +  0.5
        batch_y = batch_y +  0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs,det_xyz, det_feats, metas,batch_gt_instances_3d,lidar_query_inputs=None,):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
            ablation_mode (str): 
                - 'full': 전체 파이프라인 (LGPC   2-Stage Refinement) 실행
                - 'lgpc_only': LGPC로 오차는 예측하지만, Refinement 없이 1단계 특징을 그대로 사용
        Returns:
            list[dict]: Output results for tasks.
        """

        # ============================================================
        # 1. Original fused BEV feature
        # ============================================================

        # batch_size는 fusion_feat에 의존하지 않고
        # 입력 tensor에서 직접 얻는다.
        batch_size = inputs.shape[0]

        # Original fused BEV -> TransFusion shared conv
        fusion_feat = self.shared_conv(inputs)

        # BEV positional coordinates
        bev_pos = self.bev_pos.repeat(
            batch_size,
            1,
            1
        ).to(fusion_feat.device)

        # Flattened fused BEV:
        # IMPORTANT: still used as Decoder memory.
        fusion_feat_flatten = fusion_feat.view(
            batch_size,
            fusion_feat.shape[1],
            -1,
        )

        # ============================================================
        # 2. Select feature source used ONLY for initial query content
        #
        # Default:
        #   fused BEV
        #
        # LiDAR feasibility:
        #   LiDAR-only BEV
        # ============================================================

        # IMPORTANT:
        # Always initialize first.
        # This prevents query_source_feat from being undefined.
        query_source_feat = fusion_feat


        if self.query_source == 'lidar':

            if lidar_query_inputs is None:
                raise RuntimeError(
                    '[LiDAR Query] query_source="lidar", '
                    'but lidar_query_inputs is None.'
                )

            # lidar_query_inputs:
            # LiDAR-only SECOND + FPN output
            #
            # Apply the same shared_conv used by the original
            # TransFusion query path.
            lidar_query_feat = self.shared_conv(
                lidar_query_inputs
            )

            if lidar_query_feat.shape != fusion_feat.shape:
                raise RuntimeError(
                    '[LiDAR Query] feature shape mismatch: '
                    f'lidar={tuple(lidar_query_feat.shape)}, '
                    f'fused={tuple(fusion_feat.shape)}'
                )

            query_source_feat = lidar_query_feat


        elif self.query_source == 'fused':

            # Original behavior
            query_source_feat = fusion_feat


        else:

            raise RuntimeError(
                f'Unknown query_source="{self.query_source}". '
                'Expected "fused" or "lidar".'
            )


        # ============================================================
        # 3. SANITY CHECK
        #    MUST come AFTER query_source_feat has been assigned.
        # ============================================================

        if not hasattr(self, '_query_source_debug_done'):

            mean_abs_diff = (
                query_source_feat - fusion_feat
            ).abs().mean().item()

            print(
                "\n"
                "=========================================\n"
                "[QUERY SOURCE SANITY]\n"
                f"query_source={self.query_source}\n"
                f"inputs={tuple(inputs.shape)}\n"
                f"fusion_feat={tuple(fusion_feat.shape)}\n"
                f"query_source_feat={tuple(query_source_feat.shape)}\n"
                f"mean_abs_diff={mean_abs_diff:.6f}\n"
                "=========================================\n"
            )

            self._query_source_debug_done = True


        # ============================================================
        # 4. Flatten query source
        # ============================================================

        query_source_flatten = query_source_feat.view(
            batch_size,
            query_source_feat.shape[1],
            -1,
        )

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        # query_feat = fusion_feat_flatten.gather(
        #     index=top_proposals_index[:, None, :].expand(
        #         -1, fusion_feat_flatten.shape[1], -1),
        #     dim=-1,
        # )
        query_feat = query_source_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1,
                query_source_flatten.shape[1],
                -1,
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat  += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        ###### 시각화 하기 위한 변수 따기 ##########
        lidar_only_query_feat = query_feat.clone()

        # --- 2단계 퓨전 변수 초기화 ---
        coarse_fused_query_feat = lidar_only_query_feat # 카메라 없으면 이게 최종 입력
        refined_query_feat = lidar_only_query_feat      # 카메라 없으면 이게 최종 입력
        pred_delta_rot = torch.zeros(batch_size, 3, device=fusion_feat.device) # 기본값 0
        pred_delta_trans = torch.zeros(batch_size, 3, device=fusion_feat.device) # 기본값 0
 
        # --- 2. 카메라 정보 퓨전 (있을 경우) ---
        if det_xyz is not None and det_feats is not None:
            # --- 2a. 1단계: 거친 퓨전 ---
            bev_query_feat_stage1 = query_feat.permute(0, 2, 1) # [B, Nq, C]
            bev_query_pos_embed = self.bev_query_pos_embedding(query_pos)
            cam_proposal_feat = det_feats # [B, Nc, C]
            cam_proposal_pos_embed = self.camera_proposal_pos_embedding(det_xyz) # det_xyz는 [0,1] 정규화 상태 가정

            fused_feat_stage1 = self.fusion_cross_attention(
                query=bev_query_feat_stage1 +  bev_query_pos_embed,
                key=cam_proposal_feat +  cam_proposal_pos_embed,
                value=cam_proposal_feat
            )[0]
            bev_query_feat_stage1 = self.fusion_norm1(bev_query_feat_stage1 +  fused_feat_stage1)
            bev_query_feat_stage1 = self.fusion_norm2(bev_query_feat_stage1 +  self.fusion_ffn(bev_query_feat_stage1))
            coarse_fused_query_feat = bev_query_feat_stage1.permute(0, 2, 1).contiguous() # [B, C, Nq]

            # --- 시각화용: 중간 퓨전 결과 저장 ---
            # 카메라 퓨전이 있었다면 refined_query_feat, 없었다면 lidar_only_query_feat
            coarse_fused_query_feat_for_vis = coarse_fused_query_feat

            # # --- 2b. 캘리브레이션 오차 예측 ---
            # # LGPC Only 모드여도 오차 예측은 수행해야 함 (Rotation Loss 계산 및 검증을 위해)
            # pooled_coarse_feat = coarse_fused_query_feat.mean(dim=-1) # [B, C]
            
            # ########### old calibration network ##################
            # pred_delta_6dof = self.calibration_predictor(pooled_coarse_feat)  # [B, 6]
           
            # ######### new calibration network ####################
            # # x = pooled_coarse_feat # [B*N, calibration_input_dim]
            # # x_shared = self.calib_shared_fc(x)
            # # pred_rot = self.calib_rot_predictor(x_shared)
            # # pred_trans = self.calib_trans_predictor(x_shared)
            # # pred_delta_6dof = torch.cat([pred_rot, pred_trans], dim=-1) # [B*N, 6]

            # pred_delta_rot = pred_delta_6dof[..., :3]
            # pred_delta_trans = pred_delta_6dof[..., 3:]

            # # ==========================================================
            # # 🚀 비교 실험 분기점 (Ablation Strategy)
            # # ==========================================================
            # ablation_mode='full'
            # # --- 2c. 카메라 제안 보정 ---
            # if ablation_mode == 'full':
            #     # [Full Model]: 보정된 위치를 사용하여 2단계 정제(Refinement) 수행
            #     pc_range_tensor = torch.tensor(self.train_cfg['point_cloud_range'], device=det_xyz.device)
            #     # 보정 시에는 그래디언트 흐름 차단 가능 (오차 예측 학습에만 집중)
            #     det_xyz_corrected_norm = correct_camera_proposals(
            #         det_xyz, pred_delta_rot.detach(), pred_delta_trans.detach(), pc_range_tensor
            #     )
            #     cam_proposal_pos_embed_corrected = self.camera_proposal_pos_embedding(det_xyz_corrected_norm)

            #     # --- 2d. 2단계: 정제된 퓨전 ---
            #     refined_bev_query_feat = coarse_fused_query_feat.permute(0, 2, 1) # 1단계 결과 재사용
            #     refined_fused_feat = self.refined_attention(
            #         query=refined_bev_query_feat +  bev_query_pos_embed,
            #         key=cam_proposal_feat +  cam_proposal_pos_embed_corrected, # 보정된 위치 사용
            #         value=cam_proposal_feat
            #     )[0]
            #     refined_bev_query_feat = self.refined_norm1(refined_bev_query_feat +  refined_fused_feat)
            #     refined_bev_query_feat = self.refined_norm2(refined_bev_query_feat +  self.refined_ffn(refined_bev_query_feat))
            #     # 최종 출력: 정제된 특징
            #     refined_query_feat = refined_bev_query_feat.permute(0, 2, 1).contiguous() # [B, C, Nq]
            
            # elif ablation_mode == 'lgpc_only':
            #     # [LGPC Only]: 오차는 예측했으나(위에서 수행함), 정제(Refinement) 과정 생략
            #     # 논리: "LGPC가 오차를 알아냈다 하더라도, 이를 반영하여 
            #     #       특징맵을 다시 퓨전하지 않으면 성능 향상은 없다"는 것을 증명
                
            #     # 최종 출력: 1단계 거친 특징 (보정 전 특징)
            #     refined_query_feat = coarse_fused_query_feat
                
            # else:
            #     raise ValueError(f"Unknown ablation mode: {ablation_mode}")

            # ============================================================
            # RRRF Stage-2 strategy
            #
            # residual_se3:
            #   Original IROS RRRF
            #   coarse feature -> residual 6DoF -> geometry correction
            #   -> refined attention
            #
            # feature_refine:
            #   New RRRF-v0
            #   No Stage-2 6DoF prediction.
            #   Reuse Stage-1 LGPC-corrected proposal geometry and perform
            #   feature refinement directly.
            #
            # lgpc_only:
            #   No Stage-2 refinement.
            # ============================================================

            if self.rrrf_mode == 'residual_se3':

                # --------------------------------------------------------
                # Original Stage-2 residual SE(3) prediction
                # --------------------------------------------------------
                pooled_coarse_feat = coarse_fused_query_feat.mean(dim=-1)

                pred_delta_6dof = self.calibration_predictor(
                    pooled_coarse_feat
                )

                pred_delta_rot = pred_delta_6dof[..., :3]
                pred_delta_trans = pred_delta_6dof[..., 3:]

                # Stage-2 geometric correction
                pc_range_tensor = torch.tensor(
                    self.train_cfg['point_cloud_range'],
                    device=det_xyz.device
                )

                det_xyz_corrected_norm = correct_camera_proposals(
                    det_xyz,
                    pred_delta_rot.detach(),
                    pred_delta_trans.detach(),
                    pc_range_tensor
                )

                cam_proposal_pos_embed_refined = \
                    self.camera_proposal_pos_embedding(
                        det_xyz_corrected_norm
                    )


            elif self.rrrf_mode == 'feature_refine':

                # --------------------------------------------------------
                # NEW RRRF-v0
                #
                # IMPORTANT:
                # det_xyz has already been geometrically corrected
                # by Stage-1 LGPC before entering TransFusionHead.
                #
                # Therefore:
                #   - no residual 6DoF prediction
                #   - no correct_camera_proposals()
                #   - reuse Stage-1 corrected proposal position
                # --------------------------------------------------------

                cam_proposal_pos_embed_refined = \
                    cam_proposal_pos_embed


            elif self.rrrf_mode == 'lgpc_only':

                # No Stage-2 refinement
                refined_query_feat = coarse_fused_query_feat


            else:

                raise ValueError(
                    f'Unknown RRRF mode: {self.rrrf_mode}'
                )


            # ------------------------------------------------------------
            # Stage-2 refined feature fusion
            #
            # Used by:
            #   residual_se3
            #   feature_refine
            #
            # Not used by:
            #   lgpc_only
            # ------------------------------------------------------------
            if self.rrrf_mode in [
                'residual_se3',
                'feature_refine'
            ]:

                refined_bev_query_feat = \
                    coarse_fused_query_feat.permute(
                        0, 2, 1
                    )

                refined_fused_feat = self.refined_attention(
                    query=(
                        refined_bev_query_feat
                        + bev_query_pos_embed
                    ),

                    key=(
                        cam_proposal_feat
                        + cam_proposal_pos_embed_refined
                    ),

                    value=cam_proposal_feat
                )[0]

                refined_bev_query_feat = self.refined_norm1(
                    refined_bev_query_feat
                    + refined_fused_feat
                )

                refined_bev_query_feat = self.refined_norm2(
                    refined_bev_query_feat
                    + self.refined_ffn(
                        refined_bev_query_feat
                    )
                )

                refined_query_feat = \
                    refined_bev_query_feat.permute(
                        0, 2, 1
                    ).contiguous()
            
        
        # --- 시각화용: 중간 퓨전 결과 저장 ---
        # 카메라 퓨전이 있었다면 refined_query_feat, 없었다면 lidar_only_query_feat
        fused_query_feat_for_vis = refined_query_feat 

        # --- 3. Transformer Decoder 정제 ---
        decoder_input_feat = refined_query_feat # 퓨전 결과가 디코더 입력
        
        ret_dicts = []
        current_query_pos = query_pos # 초기 위치로 시작
        for i in range(self.num_decoder_layers):
            decoder_output_feat = self.decoder[i](
                decoder_input_feat,
                key=fusion_feat_flatten,
                query_pos=current_query_pos,
                key_pos=bev_pos
            )
            
            res_layer = self.prediction_heads[i](decoder_output_feat)
            predicted_center = res_layer['center'] + current_query_pos.permute(0, 2, 1)
            res_layer['center'] = predicted_center
            ret_dicts.append(res_layer)

            # 다음 레이어를 위해 특징과 위치 업데이트
            decoder_input_feat = decoder_output_feat 
            current_query_pos = predicted_center.detach().clone().permute(0, 2, 1)

        # --- 시각화용: 최종 디코더 출력 특징 저장 ---
        final_query_feat_for_vis = decoder_output_feat

        # # --- 4. 시각화 호출 ---
        # if self.training_step % 3000 == 0 :
        #     with torch.no_grad():
        #         gt_instances_3d = batch_gt_instances_3d[0] # forward_single은 배치 0만 처리 가정
        #         pc_range = self.train_cfg['point_cloud_range']
        #         voxel_size = self.train_cfg['voxel_size']
                
        #         # det_xyz, det_feats가 None일 경우 빈 텐서 전달 (오류 방지)
        #         vis_cam_xyz = det_xyz[0] if det_xyz is not None else torch.empty(0, 3, device=decoder_output_feat.device)
        #         vis_cam_feat = det_feats[0] if det_feats is not None else torch.empty(0, lidar_only_query_feat.shape[1], device=decoder_output_feat.device)
                
        #         visualize_full_pipeline_enhanced(
        #             cam_proposals_xyz=vis_cam_xyz,
        #             cam_proposals_feat=vis_cam_feat,
        #             query_pos=query_pos[0], # 항상 초기 위치 전달
        #             lidar_only_feat=lidar_only_query_feat[0].permute(1, 0),
        #             coarse_fused_feat=coarse_fused_query_feat_for_vis[0].permute(1, 0),
        #             fused_feat=fused_query_feat_for_vis[0].permute(1, 0),
        #             final_feat=final_query_feat_for_vis[0].permute(1, 0),
        #             gt_bboxes_3d=gt_instances_3d.bboxes_3d,
        #             pc_range=pc_range,
        #             voxel_size=voxel_size,
        #             step=self.training_step,
        #             save_path=f"work_dirs/full_pipeline_step_{self.training_step}.png"
        #         )
        # self.training_step  = 1

        # --- 5. 결과 처리 및 반환 ---
        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1
        )
        ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
             # 마지막 레이어 결과만 반환 시, pred_delta_* 추가 필요
             last_res = ret_dicts[-1]
             last_res['pred_delta_rot'] = pred_delta_rot
             last_res['pred_delta_trans'] = pred_delta_trans
             return [last_res]

        # 모든 레이어 결과 반환 시, pred_delta_* 추가 필요
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        
        # ✨ 추가: 2단계 퓨전 결과(디코더 입력 전)를 loss 계산용으로 전달
        new_res['refined_query_feat'] = refined_query_feat
        
        new_res['pred_delta_rot'] = pred_delta_rot
        new_res['pred_delta_trans'] = pred_delta_trans
        
        return [new_res]

    # def forward(self, feats, metas):
    #     """Forward pass.

    #     Args:
    #         feats (list[torch.Tensor]): Multi-level features, e.g.,
    #             features produced by FPN.
    #     Returns:
    #         tuple(list[dict]): Output results. first index by level, second
    #         index by layer
    #     """
    #     if isinstance(feats, torch.Tensor):
    #         feats = [feats]
    #     res = multi_apply(self.forward_single, feats, [metas])
    #     assert len(res) == 1, 'only support one level features.'
    #     return res
    
    def forward(self, feats, det_xyz=None, det_feats=None, metas=None,batch_gt_instances_3d=None,lidar_query_feats=None,):
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        if lidar_query_feats is None:

            lidar_query_feats = [
                None
                for _ in range(len(feats))
            ]

        elif isinstance(
            lidar_query_feats,
            torch.Tensor
        ):

            lidar_query_feats = [
                lidar_query_feats
            ]

        else:

            lidar_query_feats = list(
                lidar_query_feats
            )

        assert len(lidar_query_feats) == len(feats)
        # multi_apply 호출 시에도 순서만 맞춰주면 됩니다.
        res = multi_apply(self.forward_single, feats, [det_xyz], [det_feats], [metas],[batch_gt_instances_3d],lidar_query_feats,)
        
        assert len(res) == 1, 'only support one level features.'
        # return res
        return res

    # def predict(self, batch_feats, batch_input_metas):
    #     preds_dicts = self(batch_feats, batch_input_metas)
    #     res = self.predict_by_feat(preds_dicts, batch_input_metas)
    #     return res

    def predict(self, batch_feats, det_xyz, det_feats, batch_input_metas,lidar_query_feats=None,):
        # self()는 forward를 호출. 이제 모든 인자를 올바르게 전달합니다.
        preds_dicts = self(batch_feats, det_xyz, det_feats, batch_input_metas,lidar_query_feats=lidar_query_feats)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores,dtype=torch.bool)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores,dtype=torch.bool)
                        for cls_idx in task['indices']:
                            task_mask  = labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            # keep_indices = torch.where(
                            #     task_mask != 0)[0][task_keep_indices]
                            # keep_mask[keep_indices] = 1
                             keep_indices = torch.where(task_mask)[0][task_keep_indices.to(scores.device)]
                             keep_mask[keep_indices] = True
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                # temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                temp_instances.bboxes_3d = metas[i]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                # # ===================== [HERE] stage1 값을 stage2에서 읽는 위치 =====================
                # # metas == img_metas (배치별 dict 리스트)
                # stage1_rot = metas[i].get('stage1_pred_delta_rot', None)      # (N_cam, 3) 기대
                # stage1_trans = metas[i].get('stage1_pred_delta_trans', None)  # (N_cam, 3) 기대
                # active = metas[i].get('stage1_active_cam_indices', None)
                # # ================================================================================

                if self.rrrf_mode == 'residual_se3':

                    pred_rot = preds_dict[0].get(
                        'pred_delta_rot',
                        None
                    )

                    pred_trans = preds_dict[0].get(
                        'pred_delta_trans',
                        None
                    )

                else:

                    # No explicit Stage-2 calibration prediction
                    pred_rot = None
                    pred_trans = None
                # ===================== [NEW] 캘리브레이션 예측값도 같이 저장 =====================
                # preds_dict[0]는 dict, batch 차원은 i
                # 수정 (OK): metainfo로 저장 (길이 체크 없음)
                if pred_rot is not None and pred_trans is not None:
                    # temp_instances.set_metainfo(dict(
                    #     pred_delta_rot=pred_rot[i].detach().cpu(),
                    #     pred_delta_trans=pred_trans[i].detach().cpu(),
                    # ))
                    metas[i]['pred_delta_rot_2nd'] = pred_rot[i].detach().cpu()
                    metas[i]['pred_delta_trans_2nd'] = pred_trans[i].detach().cpu()
                # ===============================================================================

                # ===================== [COMPLETE] stage1   stage2(잔차) 합성 & 저장 =====================
                def _as_cpu_tensor(x):
                    if x is None:
                        return None
                    if torch.is_tensor(x):
                        return x.detach().cpu()
                    return torch.tensor(x).detach().cpu()

                # --- stage1 읽기 (카메라별) ---
                stage1_rot_cam = _as_cpu_tensor(metas[i].get('stage1_pred_delta_rot', None))      # (N_cam,3)
                stage1_trans_cam = _as_cpu_tensor(metas[i].get('stage1_pred_delta_trans', None))  # (N_cam,3)
                active = metas[i].get('stage1_active_cam_indices', None)
                active_mask = metas[i].get('stage1_active_cam_mask', None)
                active = _as_cpu_tensor(active) if active is not None else None
                active_mask = _as_cpu_tensor(active_mask).bool() if active_mask is not None else None

                # --- stage2 읽기 (residual, 배치별 1개) ---
                pred2_rot = _as_cpu_tensor(pred_rot[i]) if pred_rot is not None else None         # (3,)
                pred2_trans = _as_cpu_tensor(pred_trans[i]) if pred_trans is not None else None   # (3,)

                # --- stage1 mean 계산 (active만 평균내는 게 물리적으로 가장 정확) ---
                stage1_rot_mean = None
                stage1_trans_mean = None
                if stage1_rot_cam is not None and stage1_trans_cam is not None:
                    if active_mask is not None and active_mask.any():
                        stage1_rot_mean = stage1_rot_cam[active_mask].mean(dim=0)     # (3,)
                        stage1_trans_mean = stage1_trans_cam[active_mask].mean(dim=0) # (3,)
                    elif active is not None and active.numel() > 0:
                        act_idx = active.long().view(-1)
                        stage1_rot_mean = stage1_rot_cam[act_idx].mean(dim=0)
                        stage1_trans_mean = stage1_trans_cam[act_idx].mean(dim=0)
                    else:
                        # active 정보 없으면 전체 평균(차선책)
                        stage1_rot_mean = stage1_rot_cam.mean(dim=0)
                        stage1_trans_mean = stage1_trans_cam.mean(dim=0)

                    # metric 호환 alias도 metas에 같이 심어둠 (선택이지만 추천)
                    metas[i]['pred_delta_rot_1st'] = stage1_rot_cam
                    metas[i]['pred_delta_trans_1st'] = stage1_trans_cam

                # --- 최종 합성 (R_total = R2@R1, t_total = t1 t2) ---
                total_R = None
                total_t = None
                if (stage1_rot_mean is not None) and (pred2_rot is not None):
                    R1 = axis_angle_to_matrix(stage1_rot_mean[None].float())   # (1,3,3)
                    R2 = axis_angle_to_matrix(pred2_rot[None].float())         # (1,3,3)
                    total_R = (R2 @ R1).squeeze(0)                              # (3,3)
                if (stage1_trans_mean is not None) and (pred2_trans is not None):
                    total_t = (stage1_trans_mean.float() + pred2_trans.float()) # (3,)

                # --- 저장(InstanceData.metainfo): stage2 residual   stage1   total ---
                meta_payload = {}
                if pred2_rot is not None and pred2_trans is not None:
                    meta_payload.update(dict(
                        # ✅ metric이 찾는 키들 (권장)
                        pred_delta_rot_2nd=pred2_rot,
                        pred_delta_trans_2nd=pred2_trans,
                        stage2_pred_delta_rot=pred2_rot,
                        stage2_pred_delta_trans=pred2_trans,

                        # ✅ 기존 호환 (너 코드/다른 모듈이 쓰는 키)
                        pred_delta_rot=pred2_rot,
                        pred_delta_trans=pred2_trans,
                    ))
                if stage1_rot_cam is not None and stage1_trans_cam is not None:
                    meta_payload.update(dict(
                        stage1_pred_delta_rot=stage1_rot_cam,       # (N_cam,3)
                        stage1_pred_delta_trans=stage1_trans_cam,
                        stage1_pred_delta_rot_mean=stage1_rot_mean,  # (3,)
                        stage1_pred_delta_trans_mean=stage1_trans_mean,
                    ))
                if active is not None:
                    meta_payload['stage1_active_cam_indices'] = active.long()
                if active_mask is not None:
                    meta_payload['stage1_active_cam_mask'] = active_mask
                if total_R is not None:
                    meta_payload['pred_delta_rot_total_R'] = total_R
                if total_t is not None:
                    meta_payload['pred_delta_trans_total'] = total_t

                if len(meta_payload) > 0:
                    temp_instances.set_metainfo(meta_payload)
                # ===================== [END COMPLETE] ================================================

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels  += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, det_xyz,det_feats, batch_data_samples,pred_delta_rot=None, pred_delta_trans=None, gt_delta_rot=None, gt_delta_trans=None):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats,det_xyz, det_feats,batch_input_metas,batch_gt_instances_3d)
        loss = self.loss_by_feat(
                    preds_dicts, 
                    batch_gt_instances_3d, 
                    batch_input_metas, # metas는 여전히 다른 용도로 필요할 수 있음
                    pred_delta_rot=pred_delta_rot,
                    pred_delta_trans=pred_delta_trans,
                    gt_delta_rot=gt_delta_rot,       # <-- 전달
                    gt_delta_trans=gt_delta_trans   # <-- 전달
                )
        
        # --- ✨ 3. 시각화를 위해 예측값 추가 ---
        # forward_single이 배치 0만 처리한다고 가정하지 않고,
        # 전체 배치를 처리한 preds_list[0] (new_res 딕셔너리)를 사용합니다.
        # 이 딕셔너리는 [B, ...] 차원의 텐서를 포함합니다.
        
        # new_res 딕셔너리 (preds_list[0])에서 예측값을 가져와 'losses'에 추가
        new_res = preds_dicts[0][0] 
        loss['pred_delta_rot'] = new_res['pred_delta_rot']
        loss['pred_delta_trans'] = new_res['pred_delta_trans']
        # ------------------------------------

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData],
                     batch_input_metas: List[dict],
                     pred_delta_rot: torch.Tensor,    # <-- 추가
                     pred_delta_trans: torch.Tensor,
                     gt_delta_rot: torch.Tensor,    # <-- 추가
                     gt_delta_trans: torch.Tensor,
                     *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # # --- ✨ 추가: 캘리브레이션 오차 예측 Loss 계산 ✨ ---
        # # forward_single에서 반환된 예측값 사용
        # pred_delta_rot_2nd = preds_dict['pred_delta_rot']
        # pred_delta_trans_2nd = preds_dict['pred_delta_trans']

        # # 1단계 예측값 (detach()로 그래디언트 차단)
        # pred_delta_rot_1st_per_cam = pred_delta_rot.detach()
        # pred_delta_trans_1st_per_cam = pred_delta_trans.detach()

        # # [수정] 1단계 예측값도 6개 카메라에 대해 평균을 냅니다.
        # pred_delta_rot_1st_mean = pred_delta_rot_1st_per_cam.mean(dim=1) # Shape [1, 3]
        # pred_delta_trans_1st_mean = pred_delta_trans_1st_per_cam.mean(dim=1) # Shape [1, 3]

        # # 전체 GT
        # gt_delta_rot_mean = gt_delta_rot.mean(dim=1)
        # gt_delta_trans_mean = gt_delta_trans.mean(dim=1)
        
        # # Loss 계산 (배치 전체에 대해 mean)
        # R_pred_1st = axis_angle_to_matrix(pred_delta_rot_1st_mean)
        # R_gt_total = axis_angle_to_matrix(gt_delta_rot_mean)
        # R_gt_residual = R_gt_total @ R_pred_1st.transpose(1, 2)
        # # T_gt_residual = T_gt_total - T_pred_1st
        # T_gt_residual = gt_delta_trans_mean - pred_delta_trans_1st_mean

        # R_pred_2nd = axis_angle_to_matrix(pred_delta_rot_2nd)

        # loss_calib_rot_pred= identity_matrix_loss(R_pred_2nd, R_gt_residual)
        # loss_calib_trans_pred = F.smooth_l1_loss(pred_delta_trans_2nd, T_gt_residual, reduction='mean')

        # loss_dict['loss_calib_rot_pred'] = loss_calib_rot_pred * 100.0 # 가중치
        # loss_dict['loss_calib_trans_pred'] = loss_calib_trans_pred * 50.0 # 가중치
        # ----------------------------------------------------

        # ============================================================
        # Stage-2 residual SE(3) supervision
        # ============================================================

        if self.rrrf_mode == 'residual_se3':

            pred_delta_rot_2nd = \
                preds_dict['pred_delta_rot']

            pred_delta_trans_2nd = \
                preds_dict['pred_delta_trans']

            # Stage-1 LGPC prediction
            pred_delta_rot_1st_per_cam = \
                pred_delta_rot.detach()

            pred_delta_trans_1st_per_cam = \
                pred_delta_trans.detach()

            pred_delta_rot_1st_mean = \
                pred_delta_rot_1st_per_cam.mean(dim=1)

            pred_delta_trans_1st_mean = \
                pred_delta_trans_1st_per_cam.mean(dim=1)

            # GT
            gt_delta_rot_mean = \
                gt_delta_rot.mean(dim=1)

            gt_delta_trans_mean = \
                gt_delta_trans.mean(dim=1)

            # Residual rotation target
            R_pred_1st = axis_angle_to_matrix(
                pred_delta_rot_1st_mean
            )

            R_gt_total = axis_angle_to_matrix(
                gt_delta_rot_mean
            )

            R_gt_residual = (
                R_gt_total
                @ R_pred_1st.transpose(1, 2)
            )

            # Residual translation target
            T_gt_residual = (
                gt_delta_trans_mean
                - pred_delta_trans_1st_mean
            )

            R_pred_2nd = axis_angle_to_matrix(
                pred_delta_rot_2nd
            )

            loss_calib_rot_pred = \
                identity_matrix_loss(
                    R_pred_2nd,
                    R_gt_residual
                )

            loss_calib_trans_pred = \
                F.smooth_l1_loss(
                    pred_delta_trans_2nd,
                    T_gt_residual,
                    reduction='mean'
                )

            loss_dict['loss_calib_rot_pred'] = \
                loss_calib_rot_pred * 100.0

            loss_dict['loss_calib_trans_pred'] = \
                loss_calib_trans_pred * 50.0

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # --- ✨ 추가: 2단계 정제 퓨전(Refined Fusion)에 대한 보조 Loss 계산 ✨ ---
        # forward_single에서 전달받은 2단계 퓨전 특징
        refined_query_feat = preds_dict['refined_query_feat']
        
        # 보조 예측 헤드로 예측 수행
        aux_preds = self.refined_fusion_aux_head(refined_query_feat)
        
        # 보조 Loss 계산 (디코더 루프의 마지막 레이어(layer_-1)와 동일한 방식 사용)
        aux_prefix = 'layer_refined_fusion'
        num_layer_proposals = self.num_proposals # 보조 헤드는 디코더처럼 누적되지 않음
        
        aux_labels = labels[..., -num_layer_proposals:].reshape(-1)
        aux_label_weights = label_weights[..., -num_layer_proposals:].reshape(-1)
        aux_cls_score = aux_preds['heatmap'].permute(0, 2, 1).reshape(-1, self.num_classes)
        
        loss_aux_cls = self.loss_cls(
            aux_cls_score.float(),
            aux_labels,
            aux_label_weights,
            avg_factor=max(num_pos, 1),
        )

        aux_center = aux_preds['center']
        aux_height = aux_preds['height']
        aux_rot = aux_preds['rot']
        aux_dim = aux_preds['dim']
        aux_preds_tensor = torch.cat([aux_center, aux_height, aux_dim, aux_rot], dim=1).permute(0, 2, 1)
        
        if 'vel' in aux_preds:
            aux_vel = aux_preds['vel']
            aux_preds_tensor = torch.cat([aux_center, aux_height, aux_dim, aux_rot, aux_vel], dim=1).permute(0, 2, 1)

        code_weights = self.train_cfg.get('code_weights', None)
        aux_bbox_weights = bbox_weights[:, -num_layer_proposals:, :]
        aux_reg_weights = aux_bbox_weights * aux_bbox_weights.new_tensor(code_weights)
        aux_bbox_targets = bbox_targets[:, -num_layer_proposals:, :]
        
        loss_aux_bbox = self.loss_bbox(
            aux_preds_tensor,
            aux_bbox_targets,
            aux_reg_weights,
            avg_factor=max(num_pos, 1)
        )

        loss_dict[f'{aux_prefix}_loss_cls'] = loss_aux_cls * 0.5   # 가중치 (예: 0.5)
        loss_dict[f'{aux_prefix}_loss_bbox'] = loss_aux_bbox * 0.5 # 가중치 (예: 0.5)
        # ----------------------------------------------------

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +  
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer + 
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict
