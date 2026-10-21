
# ============================================================
# EXP5 - RRRF V2 Adapter-Only Fine-Tuning
# Base: EXP4-B Joint 83K
#
# No changes to bevfusion.py / transfusion_head.py
# Uses EXISTING stage: rrrf_adapter
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp4_joint_fresh_train.py'
]

model = dict(
    lgpc_train_stage='rrrf_adapter',

    enable_selective_freezing=False,
    calibration_mode='pcc_full',

    bbox_head=dict(
        rrrf_mode='residual_se3_v2',

        # Geometry OFF
        # Keep module instantiated for checkpoint compatibility.
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=0.0,

        # Adapter ON - only trainable module
        rrrf_adapter_enabled=True,
        rrrf_adapter_scale=1.0,
    ),
)

# New training starts at iteration 0
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=10000,
    val_interval=999999,
)

# Reset Exp4 optimizer settings.
# Exp4 Adapter used lr_mult=2.0.
# Here actual Adapter LR = 1e-5.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=0.01,
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)

param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=8,
        save_last=True,
    ),
)

# 83K model weights only
load_from = (
    'work_dirs/pcc_exp4_joint_fresh_400k/iter_83000.pth'
)

# Do not resume previous optimizer
resume = False
