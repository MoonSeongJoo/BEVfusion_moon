# ============================================================
# LGPC Stage-1 Camera-Pose-Conditioned CalibHead
# 500-sample calibration evaluation
#
# Evaluation target:
#
#   CorrNet     : pretrained / eval
#   ZEstimator  : pretrained / eval
#   CalibHead   : camera-pose-conditioned checkpoint / eval
#   RRRF        : NOT evaluated
#
# IMPORTANT:
#   pcc_calib_only
#   => REAL CalibHead prediction is evaluated.
#
#   DO NOT use geo_oracle_gtrot here.
# ============================================================


_base_ = [
    './pcc_bevfusion_repair_v1_full24.py'
]


# ============================================================
# MODEL
# ============================================================

model = dict(

    enable_selective_freezing=False,

    # ========================================================
    # CRITICAL:
    # Test actual learned LGPC CalibHead.
    # ========================================================
    calibration_mode='pcc_calib_only',

    # Stage-1 CalibHead architecture
    lgpc_train_stage='calib',

    bbox_head=dict(
        rrrf_mode='lgpc_only',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',
    ),

    corr=dict(
        init_cfg=None,
        enable_cycle=False,
    ),

    z_estimator=dict(
        init_cfg=None,
    ),

    calib_head=dict(
        type='CalibrationCorrectionHead',

        in_channels=312,
        num_kp=200,

        dropout_p=0.1,
        local_dim=32,
        point_dim=64,
        global_dim=64,

        rot_condition_dim=16,
        rot_condition_scale_deg=10.0,

        # ====================================================
        # Camera-pose conditioning
        # Must match the trained checkpoint.
        # ====================================================
        camera_pose_dim=16,
        # ====================================================
        # Z source ablation
        # learned / raw_fallback / pred_only
        # ====================================================
        z_gate_mode='learned',

        init_cfg=None,
    ),
)


# ============================================================
# TEST LOOP ONLY
# ============================================================

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)


# ============================================================
# Calibration Metric
# ============================================================

test_evaluator = [

    dict(
        type='CalibLCCNetMetric',

        # Change prefix so it cannot be confused
        # with geo_oracle_gtrot results.
        prefix='cam_embed',

        debug=True,
        debug_n=5,
    ),
]


# ============================================================
# IMPORTANT:
# tools/test.py positional checkpoint is used.
#
# Avoid confusing this evaluation with an old load_from.
# ============================================================

load_from = None
resume = False