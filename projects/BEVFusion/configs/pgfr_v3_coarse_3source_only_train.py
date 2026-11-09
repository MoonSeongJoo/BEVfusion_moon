# ============================================================
# PGFR-Det Ver3.0
# Coarse Fusion Only Fine-tuning
# ============================================================

_base_ = ['./pgfr_coarse_3source_train.py']


# ============================================================
# 1. Model
# ============================================================

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_coarse_only',

    # CorrNet BEV branch
    use_corr_aligned_bev=True,

    corr_bev_feature_scale=1.0,

    bbox_head=dict(

        rrrf_mode='feature_refine',

        query_source='fused',

        # Coarse ON
        rrrf_use_coarse=True,

        # Refined runs but stays frozen
        rrrf_use_refined=True,

        # 3-source Key/Value
        rrrf_coarse_dual_enabled=True,

        rrrf_coarse_dual_bev_channels=160,

        rrrf_coarse_dual_scale=1.0,

        # Query Selector OFF
        rrrf_query_select_enabled=False,

        rrrf_query_select_scale=0.0,

        # Residual Adapter OFF
        rrrf_adapter_enabled=False,

        rrrf_adapter_scale=0.0,
    ),
)


# ============================================================
# 2. Pretrained Checkpoint
# ============================================================

load_from = (
    '/workspace/mmdetection3d/data/weights/'
    'PGFR_Det_final_model.pth'
)

# Model weights only, not optimizer state
resume = False

# ============================================================
# 3. Training + Final Validation
# ============================================================

# ============================================================
# 3. Training + Final Validation
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=20000,
)

# Enable validation
val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

# IMPORTANT: Ignore inherited val_dataloader=None
val_dataloader = dict(
    _delete_=True,
    **_base_.test_dataloader
)

# Reuse existing test metrics
val_evaluator = _base_.test_evaluator


# ============================================================
# 4. DataLoader
# ============================================================

train_dataloader = dict(

    batch_size=1,

    num_workers=12,

    persistent_workers=False,
)


# ============================================================
# 5. Optimizer
# ============================================================

optim_wrapper = dict(

    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(

        type='AdamW',

        lr=1e-5,

        weight_decay=0.01,
    ),

    paramwise_cfg=dict(

        custom_keys={

            # CorrNet BEV Lifter
            'corr_bev_lifter': dict(
                lr_mult=5.0,
            ),

            # 3-source BEV Adapter
            'bbox_head.coarse_bev_adapter': dict(
                lr_mult=5.0,
            ),

            # Coarse Gate
            'bbox_head.coarse_bev_gate': dict(
                lr_mult=5.0,
            ),

            # Residual Alpha
            'bbox_head.coarse_bev_alpha': dict(
                lr_mult=5.0,
                decay_mult=0.0,
            ),

            # Coarse Cross Attention
            'bbox_head.fusion_cross_attention': dict(
                lr_mult=1.0,
            ),

            # Coarse FFN
            'bbox_head.fusion_ffn': dict(
                lr_mult=1.0,
            ),

            # Coarse Normalization
            'bbox_head.fusion_norm1': dict(
                lr_mult=1.0,
            ),

            'bbox_head.fusion_norm2': dict(
                lr_mult=1.0,
            ),
        }
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)


# ============================================================
# 6. Scheduler
# ============================================================

param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)


# ============================================================
# 7. DDP
# ============================================================

find_unused_parameters = True


# ============================================================
# 8. Logging / Checkpoint
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
# 9. Work Directory
# ============================================================

work_dir = (
    'work_dirs/pgfr_v3_coarse_3source_only_2k'
)