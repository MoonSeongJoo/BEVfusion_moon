# ============================================================
# PGFR-Det
#
# No-Residual-Pose RRRF
# + Proposal Valid Mask
# + GeometrySelectiveAttention
#
# Stage-1 LGPC: Frozen
#
# Stage-2 Trainable:
#   - Geometry Selector
#   - Detection Decoder
#   - Prediction Head
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp4_joint_fresh_train.py'
]

# ------------------------------------------------------------
# 1. Model
# ------------------------------------------------------------

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_joint_fresh',

    bbox_head=dict(

        # No Stage-2 SE(3)
        rrrf_mode='feature_refine',

        rrrf_num_cams=6,

        rrrf_detach_geometry=True,

        # Keep fused BEV queries
        query_source='fused',

        # No residual feature adapter
        rrrf_adapter_enabled=False,

        rrrf_adapter_scale=0.0,

        # Enable local proposal fusion
        rrrf_query_select_enabled=True,

        rrrf_query_select_scale=1.0,

        rrrf_query_select_topk=16,

        rrrf_query_select_radius_m=4.0,

        rrrf_query_select_sigma_m=2.0,

        # No geometry ablation
        rrrf_ablation_identity_geometry=False,

        # Keep refined attention
        rrrf_ablation_skip_refined_attention=False,
    ),
)

# ------------------------------------------------------------
# 2. Training
# ------------------------------------------------------------

train_cfg = dict(

    _delete_=True,

    type='IterBasedTrainLoop',

    # Initial pilot
    max_iters=2000,

    # Full validation at end of pilot
    val_interval=2000,
)

# ------------------------------------------------------------
# 3. Optimizer
# ------------------------------------------------------------

optim_wrapper = dict(

    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1.0e-5,
        weight_decay=0.01,
    ),

    paramwise_cfg=dict(

        custom_keys={

            'bbox_head.rrrf_query_selector': dict(
                lr_mult=2.0,
            ),

            'bbox_head.decoder.0': dict(
                lr_mult=1.0,
            ),

            'bbox_head.prediction_heads.0': dict(
                lr_mult=1.0,
            ),
        },
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)

# ------------------------------------------------------------
# 4. LR Scheduler
# ------------------------------------------------------------

param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)

# ------------------------------------------------------------
# 5. Checkpoint / Logging
# ------------------------------------------------------------

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
        max_keep_ckpts=8,
        save_last=True,
    ),
)

# ------------------------------------------------------------
# 6. Load Pretrained Weights
# ------------------------------------------------------------

# Replace with a REAL, existing checkpoint.
# Prefer a verified reference with trained LGPC,
# coarse/refined RRRF and detection decoder.
#
# Do not accidentally resume optimizer state.

load_from = '/workspace/mmdetection3d/data/weights/PGFR_Det_final_model.pth'

resume = False

# ------------------------------------------------------------
# 7. Work Directory
# ------------------------------------------------------------

work_dir = (
    'work_dirs/'
    'pgfr_feature_refine_local_2k'
)