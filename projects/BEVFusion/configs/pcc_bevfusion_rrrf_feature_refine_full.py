# ============================================================
# PCC Repair V1
# LGPC Stage-2B : CalibHead-only Training
#
# Training:
#   CorrNet     : FROZEN
#   ZEstimator  : FROZEN
#   CalibHead   : FROZEN
#   RRRF        : Train
# ============================================================


_base_ = [
    './pcc_bevfusion_repair_v1_full24.py'
]


# ============================================================
# MODEL
# ============================================================

model = dict(

    enable_selective_freezing=False,
    calibration_mode= 'geo_oracle_gtrot',

    # --------------------------------------------------------
    # NEW LGPC sub-stage
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
        # No cycle forward in staged LGPC training.
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
        # NEW
        camera_pose_dim=16,
        init_cfg=None,
    ),

)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

# ============================================================
# TRAINING SCHEDULE
#
# Z-only: 4 epochs
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=4,

    # Physical calibration validation is meaningless
    # before CalibHead has been trained.
    val_interval=1,
)

# train_cfg = dict(
#     _delete_=True,
#     type='IterBasedTrainLoop',
#     max_iters=10000,
#     val_interval=999999,
# )

val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

test_evaluator = [

    dict(
        type='CalibLCCNetMetric',
        prefix='geo_gtrot',
        debug=True,
        debug_n=5,
    ),
]

val_evaluator = test_evaluator

# ============================================================
# OPTIMIZER
#
# New stage => new optimizer.
#
# Corr checkpoint weights are loaded,
# but optimizer/scheduler state is NOT resumed.
# ============================================================

# optim_wrapper = dict(
#     optimizer=dict(
#         lr=1.0e-4,
#     ),
# )

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,
    ),

    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)


# Short standalone stage:
# keep LR simple and stable.
param_scheduler = []


# ============================================================
# CHECKPOINT
# ============================================================

# default_hooks = dict(

#     logger=dict(
#         type='LoggerHook',
#         interval=50,
#     ),

#     checkpoint=dict(
#         _delete_=True,
#         type='CheckpointHook',
#         interval=1,
#         by_epoch=True,
#         max_keep_ckpts=4,
#         save_last=True,
#     ),

#     # visualization=dict(
#     #     _delete_=True,
#     #     type='Det3DVisualizationHook',
#     #     draw=False,
#     # ),
# )

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
    ),

    checkpoint=dict(
        type='CheckpointHook',

        # --------------------------------------------------
        # Save by iteration instead of epoch
        # --------------------------------------------------
        by_epoch=False,

        # Every 5000 iterations
        interval=1000,

        save_last=True,

        # Keep recent checkpoints only
        max_keep_ckpts=5,
    ),
)

# ============================================================
# DDP
# ============================================================

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

# ============================================================
# INITIALIZATION
#
# IMPORTANT:
#   load MODEL WEIGHTS from Corr Stage.
#   Do NOT resume optimizer/scheduler/epoch counter.
# ============================================================

load_from = (
    'work_dirs/'
    'pcc_bevfusion_rrrf_feature_refine_full/'
    'iter_8000.pth'
)

resume = False