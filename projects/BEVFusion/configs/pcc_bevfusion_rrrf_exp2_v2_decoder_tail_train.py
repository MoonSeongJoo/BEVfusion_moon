
# ============================================================
# PCC EXP2: RRRF V2 Decoder Tail Fine-Tuning
#
# Baseline:
#   V2 Adapter Iter3000, scale=0.0
#
# Frozen:
#   LGPC / RRRF V2 / Fusion / Adapter / Backbone
#
# Train:
#   bbox_head.decoder[0]
#   bbox_head.prediction_heads[0]
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp1_adapter_train.py'
]

model = dict(
    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_decoder_tail',

    # Match the V2 Candidate-B setup.
    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
        init_cfg=None,
    ),

    bbox_head=dict(
        rrrf_mode='residual_se3_v2',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',

        # Keep Adapter instantiated for checkpoint loading.
        # Disable its influence on Detection.
        rrrf_adapter_enabled=True,
        rrrf_adapter_scale=0.0,
    ),
)

# ============================================================
# Training: 1500 iterations
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=999999,
)

# ============================================================
# Optimizer
# ============================================================

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1.0e-5,
        weight_decay=0.01,
    ),

    # Prediction Head uses 2x base LR.
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.prediction_heads.0': dict(
                lr_mult=2.0,
            ),
        },
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)

# Constant learning rate.
param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)

# ============================================================
# Checkpoints
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
        interval=1000,
        max_keep_ckpts=8,
        save_last=True,
    ),
)

# ============================================================
# Initialize from the V2 3000-iteration checkpoint.
# ============================================================

load_from = (
    'work_dirs/pcc_exp1_v2_adapter_3k/'
    'iter_3000.pth'
)

resume = False
