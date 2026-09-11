# ============================================================
# Corr-R1 Fine-tuning
#
# Goal:
#   1) Remove repeated filler queries from Corr supervision
#   2) Pixel-balanced Huber correspondence loss
#
# Architecture:
#   SAME COTR architecture
#
# Initialization:
#   Existing trained Corr provider checkpoint
#
# Training:
#   CorrNet only
#   3000 iterations
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
    # CRITICAL
    # CorrNet only
    # --------------------------------------------------------
    lgpc_train_stage='corr',

    bbox_head=dict(
        rrrf_mode='lgpc_only',
    ),

    # --------------------------------------------------------
    # COTR architecture unchanged
    # --------------------------------------------------------
    corr=dict(
        init_cfg=None,

        # Keep cycle disabled.
        enable_cycle=False,
    ),

    # --------------------------------------------------------
    # NEW Corr-R1 loss
    #
    # Actual repeated-query removal is already implemented
    # in bevfusion.py through corr_supervision_mask.
    #
    # Here we configure only the new pixel-balanced loss.
    # --------------------------------------------------------
    corr_loss=dict(
        _delete_=True,

        type='CorrelationCycleLoss',

        corr_weight=1.0,
        cycle_weight=0.1,

        image_width=1600.0,
        image_height=900.0,

        # Huber transition at 32 original-image pixels.
        huber_beta_px=32.0,
    ),

    # Not trained in this experiment.
    z_estimator=dict(
        init_cfg=None,
    ),

    calib_head=dict(
        init_cfg=None,
    ),
)


# ============================================================
# SHORT TRAINING LOOP
#
# We do NOT need one full 61k-iter epoch.
# Stop automatically at 3000 iterations.
# ============================================================

train_cfg = dict(
    _delete_=True,

    type='IterBasedTrainLoop',

    max_iters=3000,

    # Do not launch expensive validation during this
    # initial trend experiment.
    val_interval=1000000,
)


# ============================================================
# OPTIMIZER
#
# IMPORTANT:
# Existing full24 config inherits lr=2e-4.
#
# That is too aggressive for this fine-tuning experiment.
# Start conservatively from 5e-5.
# ============================================================

optim_wrapper = dict(

    optimizer=dict(
        lr=5.0e-5,
    ),
)


# ============================================================
# LR SCHEDULER
#
# Keep LR fixed for this short diagnostic.
# Do not inherit old warmup/cosine schedule.
# ============================================================

param_scheduler = []


# ============================================================
# LOG / CHECKPOINT
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',

        # Early trend visible every 10 iter.
        interval=10,
    ),

    checkpoint=dict(
        _delete_=True,

        type='CheckpointHook',

        # IMPORTANT:
        # Save this time.
        by_epoch=False,
        interval=500,

        max_keep_ckpts=6,
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
# Use already-trained Corr provider.
#
# This checkpoint was already used as the fixed Corr/Z
# provider for the following calibration experiments.
#
# resume=False:
#   load network weights only,
#   start a NEW optimizer and NEW iteration counter.
# ============================================================

load_from = (
    'data/work_dirs/'
    'pcc_repair_v1_lgpc_zcalib_ddp2/'
    'lgpc_corr_z_only_for_calib_v3.pth'
)

resume = False