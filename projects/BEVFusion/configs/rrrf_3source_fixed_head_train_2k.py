# ============================================================
# RRRF 3-Source K/V + Fixed Official BEVFusion Head
# 2000 iterations
# ============================================================

_base_ = ['./pgfr_coarse_3source_train.py']


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_3stage_fixed_head',

    # CorrNet BEV
    use_corr_aligned_bev=True,
    corr_bev_feature_scale=1.0,

    # Load P1 as the single pretrained source
    corr=dict(init_cfg=None),
    z_estimator=dict(init_cfg=None),
    calib_head=dict(init_cfg=None),

    bbox_head=dict(

        rrrf_mode='feature_refine',

        query_source='fused',

        # Coarse Fusion
        rrrf_use_coarse=True,

        # Refine Fusion
        rrrf_use_refined=True,

        # Local Proposal Fusion
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,

        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,

        # 3-source K/V
        rrrf_coarse_dual_enabled=True,
        rrrf_coarse_dual_bev_channels=160,
        rrrf_coarse_dual_scale=1.0,

        # NEW: Apply to all RRRF stages
        rrrf_3source_all_stages=True,

        # Ablation OFF during training
        rrrf_ablation_skip_coarse_attention=False,
        rrrf_ablation_skip_refined_attention=False,
        rrrf_ablation_identity_geometry=False,

        # Disable legacy residual adapter
        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,
    ),
)


# ------------------------------------------------------------
# Pretrained checkpoint
# ------------------------------------------------------------

load_from = (
    '/workspace/mmdetection3d/data/weights/'
    'ablation_v2/PGFR_Ablation_P1.pth'
)

# Important: start a fresh optimizer
resume = False

work_dir = (
    '/workspace/mmdetection3d/work_dirs/'
    'rrrf_3source_fixed_head_2k'
)


# ------------------------------------------------------------
# Training: exactly 2000 iterations
# ------------------------------------------------------------

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=2000,
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

train_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=False,
)


# ------------------------------------------------------------
# Optimizer: RRRF trainable parameters only
# ------------------------------------------------------------

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

            'corr_bev_lifter': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_adapter': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_gate': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_alpha': dict(
                lr_mult=5.0,
                decay_mult=0.0,
            ),

            'bbox_head.rrrf_query_selector': dict(
                lr_mult=2.0,
            ),
        }
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


# ------------------------------------------------------------
# Experiment reproducibility
# ------------------------------------------------------------

find_unused_parameters = True

randomness = dict(
    seed=2026,
    deterministic=False,
    diff_rank_seed=False,
)


# ------------------------------------------------------------
# Logging and checkpoints
# ------------------------------------------------------------

default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=5,
        save_last=True,
    ),
)