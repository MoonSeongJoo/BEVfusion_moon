# ============================================================
# PCC / Hybrid IROS-70.11
# RRRF Legacy V1 2nd Calibration Predictor-Only Adaptation
#
# Goal
# ------------------------------------------------------------
# Base checkpoint:
#   hybrid_iros7011_lgpc81000_rrrf_pred_reset.pth
#
# FROZEN:
#   - Image backbone / neck / 2D detector
#   - CorrNet
#   - ZEstimator
#   - LGPC CalibHead
#   - LiDAR backbone / neck
#   - View transform
#   - Fusion layer
#   - RRRF coarse fusion modules
#   - RRRF refined fusion modules
#   - TransFusion decoder / detection heads
#
# TRAINABLE ONLY:
#   - bbox_head.calibration_predictor
#     (legacy RRRF V1 scene-global 2nd 6-DoF predictor)
#
# IMPORTANT PRECONDITION
# ------------------------------------------------------------
# bevfusion.py must already support:
#   lgpc_train_stage='rrrf_pred_only'
# and _configure_rrrf_stage()/train() must reopen ONLY
# bbox_head.calibration_predictor for this stage.
# ============================================================

_base_ = [
    './pcc_bevfusion_repair_v1_full24.py'
]


# ============================================================
# MODEL
# ============================================================

model = dict(

    # Dedicated freeze policy in BEVFusion._configure_rrrf_stage().
    enable_selective_freezing=False,

    # Use the real PCC path, not oracle modes.
    calibration_mode='pcc_full',

    # NEW dedicated stage: everything frozen except legacy
    # bbox_head.calibration_predictor.
    lgpc_train_stage='rrrf_pred_only',

    bbox_head=dict(
        # IMPORTANT: this experiment retrains the OLD / legacy RRRF V1
        # predictor, not residual_se3_v2.
        rrrf_mode='residual_se3',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',
    ),

    # Avoid nested pretrained init_cfg overriding checkpoint-loaded weights.
    corr=dict(
        init_cfg=None,
        enable_cycle=False,
    ),

    z_estimator=dict(
        init_cfg=None,
    ),

    # Explicitly match the current LGPC CalibHead architecture used by
    # the iter_81000 checkpoint.
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
        camera_pose_dim=16,
        init_cfg=None,
    ),
)


# ============================================================
# DDP
#
# Many frozen branches are present and not every parameter participates
# in loss construction, so keep find_unused_parameters=True.
# ============================================================

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)


# ============================================================
# TRAINING LOOP
#
# First controlled run: 3000 iterations.
# No automatic full 6019-frame validation during training.
# Save every 500 iters and evaluate selected checkpoints manually.
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=3000,
    val_interval=999999,
)

val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

# Make iteration logging unambiguous.
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=False,
)


# ============================================================
# EVALUATOR
#
# During predictor-only development the primary validation criterion is
# physical calibration recovery. Full NuScenes mAP/NDS will be run only
# on promising checkpoints using the dedicated full-eval config.
# ============================================================

val_evaluator = [
    dict(
        type='CalibLCCNetMetric',
        collect_device='cpu',
        prefix='calib_lccnet',
        debug=True,
        debug_n=3,
    ),
]

test_evaluator = val_evaluator


# ============================================================
# OPTIMIZER
#
# New stage = NEW optimizer state.
# Only bbox_head.calibration_predictor should have requires_grad=True.
# ============================================================

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.0e-4,
        weight_decay=0.01,
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)

# Keep the first experiment simple and easy to diagnose.
param_scheduler = []

# Do not auto-scale LR for this tiny predictor-only optimization.
auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)


# ============================================================
# HOOKS
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=7,
        save_last=True,
    ),

    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),
)


# ============================================================
# INITIALIZATION
#
# IMPORTANT:
# - This checkpoint must be the verified HYBRID checkpoint with ONLY
#   bbox_head.calibration_predictor reinitialized.
# - Do NOT resume optimizer/scheduler/iteration state.
# ============================================================

load_from = (
    'data/weights/'
    'hybrid_iros_nds7011_lgpc_iter123000.pth'
)

resume = False
