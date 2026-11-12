
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
    max_iters=20000,
    val_interval=2000,
)

val_cfg = dict(
    _delete_=True,
    type='ValLoop'
)

val_dataloader = dict(
    _delete_=True,
    **_base_.test_dataloader
)

val_evaluator = _base_.test_evaluator


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
    num_workers=12,
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
    '/workspace/mmdetection3d/data/weights/ablation_v2/PGFR_Ablation_P1.pth'
)

resume = False

work_dir = 'work_dirs/ablation_v2_B'

# ------------------------------------------------------------
# Logging / Checkpoint
# ------------------------------------------------------------

randomness = dict(
    seed=2026,
    deterministic=False,
    diff_rank_seed=False,
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

        # 매 2000 Iteration 저장
        interval=2000,

        max_keep_ckpts=5,
        save_last=True,

        # NEW: NDS 최고 성능 체크포인트 저장
        save_best='NDS',

        # NDS가 클수록 좋은 성능
        rule='greater',
    ),
)

# ============================================================
# NEW: NDS Target Early Stopping
# ============================================================

custom_hooks = [
    dict(
        type='StopOnNDSThresholdHook',

        # nuScenes Detection Score
        metric_key='NDS',

        # Target: 70.69%
        threshold=0.7069,

        # Execute AFTER CheckpointHook
        priority='LOWEST',
    ),
]