# ============================================================
# PCC / LGPC Stage-1 CalibHead Fine-Tuning
#
# Phase-A
#
# Base checkpoint:
#   work_dirs/pcc_bevfusion_rrrf_feature_refine_full/
#       iter_123000.pth
#
# Objective:
#   - Keep CorrNet frozen
#   - Keep ZEstimator frozen
#   - Train ONLY 1st Calibration Predictor (CalibHead)
#   - Keep current loss weights:
#         rotation    x100
#         translation x25
#   - Lower LR:
#         1e-4 -> 3e-5
#
# IMPORTANT:
#   load_from != resume
#
#   We load MODEL weights from iter_123000.pth,
#   but create a NEW optimizer with LR=3e-5.
# ============================================================


_base_ = [
    './pcc_bevfusion_rrrf_feature_refine_full.py'
]


# ============================================================
# MODEL
# ============================================================

model = dict(

    # Dedicated LGPC stage freeze policy is used.
    enable_selective_freezing=False,

    # Keep same physical-calibration training path.
    calibration_mode='geo_oracle_gtrot',

    # --------------------------------------------------------
    # IMPORTANT
    #
    # calib:
    #   CorrNet     = FROZEN
    #   ZEstimator  = FROZEN
    #   CalibHead   = TRAIN
    #   BEVFusion   = FROZEN
    #   RRRF        = FROZEN
    # --------------------------------------------------------
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

        camera_pose_dim=16,

        init_cfg=None,
    ),
)


# ============================================================
# DDP
# ============================================================

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',

    # lgpc_only does not necessarily use every parameter.
    find_unused_parameters=True,
)


# ============================================================
# TRAIN LOOP
#
# Controlled fine-tuning instead of another full epoch.
#
# Evaluate checkpoints:
#   iter_500
#   iter_1000
#   ...
#   iter_4000
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=3000,
    # Do not perform expensive full validation while training.
    # We evaluate physical calibration separately.
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


# ============================================================
# LOGGING
# ============================================================

log_processor = dict(
    type='LogProcessor',

    window_size=50,

    by_epoch=False,
)


# ============================================================
# PHYSICAL CALIBRATION EVALUATOR
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
# IMPORTANT:
#
# Existing:
#       LR = 1e-4
#
# Phase-A:
#       LR = 3e-5
#
# This is intentional fine-tuning.
# ============================================================

optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',

        lr=3.0e-5,

        weight_decay=0.01,
    ),

    clip_grad=dict(
        max_norm=35,

        norm_type=2,
    ),
)


# No LR scheduler in this short fine-tuning phase.
param_scheduler = []


# ============================================================
# AUTO LR SCALE
#
# Explicitly disable because we manually selected LR=3e-5.
# ============================================================

auto_scale_lr = dict(
    enable=False,

    base_batch_size=32,
)


# ============================================================
# CHECKPOINT / LOG HOOKS
#
# checkpoint every 500 iterations
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',

        interval=50,
    ),

    checkpoint=dict(
        _delete_=True,

        type='CheckpointHook',

        by_epoch=False,

        interval=500,

        max_keep_ckpts=10,

        save_last=True,
    ),

    visualization=dict(
        _delete_=True,

        type='Det3DVisualizationHook',

        draw=False,
    ),
)


# ============================================================
# INITIAL CHECKPOINT
#
# IMPORTANT:
#
# load_from:
#     model parameter initialization
#
# resume=False:
#     DO NOT restore old optimizer
#     DO NOT restore old LR
#     DO NOT continue old iteration counter
#
# Therefore:
#
# iter_123000.pth
#       ↓ model weight
#
# NEW AdamW
# LR = 3e-5
# iteration starts from 0
# ============================================================

load_from = (
    'work_dirs/'
    'pcc_bevfusion_lgpc_calib_ft123k_A/'
    'iter_4000.pth'
)

resume = False