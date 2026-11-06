
# ============================================================
# PGFR-Det Ver2.0
# Ablation B: Decoder/Prediction Head Fine-Tuning Only
# ============================================================

_base_ = ['./pgfr_feature_refine_local_train.py']

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_ablation',

    bbox_head=dict(

        rrrf_mode='feature_refine',

        query_source='fused',

        rrrf_num_cams=6,

        rrrf_detach_geometry=True,

        # Coarse OFF
        rrrf_use_coarse=False,

        # Refined OFF
        rrrf_use_refined=False,

        # Local OFF
        rrrf_query_select_enabled=False,
        rrrf_query_select_scale=0.0,

        # Adapter OFF
        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,

        # Legacy ablation OFF
        rrrf_ablation_identity_geometry=False,
        rrrf_ablation_skip_refined_attention=False,
    ),
)

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=81000,
    val_interval=999999,
)

val_cfg = None
val_dataloader = None
val_evaluator = None

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,
    ),

    paramwise_cfg=dict(
        custom_keys={

            # Projector: only C/D/E trainable
            'feat_projector': dict(lr_mult=2.0),

            # Coarse
            'bbox_head.fusion_cross_attention':
                dict(lr_mult=2.0),
            'bbox_head.fusion_ffn':
                dict(lr_mult=2.0),
            'bbox_head.fusion_norm1':
                dict(lr_mult=2.0),
            'bbox_head.fusion_norm2':
                dict(lr_mult=2.0),
            'bbox_head.camera_proposal_pos_embedding':
                dict(lr_mult=2.0),
            'bbox_head.bev_query_pos_embedding':
                dict(lr_mult=2.0),

            # Refined
            'bbox_head.refined_attention':
                dict(lr_mult=2.0),
            'bbox_head.refined_ffn':
                dict(lr_mult=2.0),
            'bbox_head.refined_norm1':
                dict(lr_mult=2.0),
            'bbox_head.refined_norm2':
                dict(lr_mult=2.0),

            # Local Proposal
            'bbox_head.rrrf_query_selector':
                dict(lr_mult=2.0),

            # Official Decoder and Prediction Head
            'bbox_head.decoder.0':
                dict(lr_mult=1.0),
            'bbox_head.prediction_heads.0':
                dict(lr_mult=1.0),
        },
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)

param_scheduler = []

auto_scale_lr = dict(enable=False, base_batch_size=32)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

randomness = dict(
    seed=20261106,
    deterministic=False,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=False,
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

# ------------------------------------------------------------
# Common P0 Checkpoint
# ------------------------------------------------------------

load_from = (
    'data/weights/ablation_v2/'
    'PGFR_Ablation_P0.pth'
)

resume = False

work_dir = 'work_dirs/ablation_v2_B'

# ------------------------------------------------------------
# Logging / Checkpoint
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
        interval=2000,
        save_last=True,
        max_keep_ckpts=12,
    ),
)
