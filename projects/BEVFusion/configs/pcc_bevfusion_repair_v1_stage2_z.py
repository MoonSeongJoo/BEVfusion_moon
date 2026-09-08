# ============================================================
# PCC Repair V1
# LGPC Stage-2A : ZEstimator-only Training
#
# Input checkpoint:
#   CorrNet Stage-1 epoch 10
#
# Training:
#   CorrNet     : FROZEN + no_grad
#   ZEstimator  : TRAIN
#   CalibHead   : FROZEN / not executed
#
# Validation:
#   disabled during Z-only stage
# ============================================================


_base_ = [
    './pcc_bevfusion_repair_v1_full24.py'
]


# ============================================================
# MODEL
# ============================================================

model = dict(

    enable_selective_freezing=False,

    # --------------------------------------------------------
    # NEW LGPC sub-stage
    # --------------------------------------------------------
    lgpc_train_stage='z',

    bbox_head=dict(
        rrrf_mode='lgpc_only',
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
        init_cfg=None,
    ),
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
    val_interval=999,
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
# OPTIMIZER
#
# New stage => new optimizer.
#
# Corr checkpoint weights are loaded,
# but optimizer/scheduler state is NOT resumed.
# ============================================================

optim_wrapper = dict(
    optimizer=dict(
        lr=1.0e-4,
    ),
)


# Short standalone stage:
# keep LR simple and stable.
param_scheduler = []


# ============================================================
# CHECKPOINT
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=4,
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
#   load MODEL WEIGHTS from Corr Stage.
#   Do NOT resume optimizer/scheduler/epoch counter.
# ============================================================

load_from = (
    'data/work_dirs/'
    'pcc_repair_v1_lgpc_full24_ddp2/'
    'epoch_10.pth'
)

resume = False