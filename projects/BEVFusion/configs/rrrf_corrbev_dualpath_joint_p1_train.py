_base_ = [
    './rrrf_corrbev_primary_fixed_head_train_2k.py'
]

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    # NEW Training Stage
    lgpc_train_stage='rrrf_dualpath_joint',

    use_corr_aligned_bev=True,

    corr_bev_feature_scale=1.0,

    bbox_head=dict(

        rrrf_mode='feature_refine',

        query_source='fused',

        rrrf_use_coarse=True,
        rrrf_use_refined=True,

        rrrf_query_select_enabled=True,

        rrrf_coarse_dual_enabled=True,

        rrrf_coarse_dual_bev_channels=160,

        rrrf_3source_all_stages=True,

        rrrf_corrbev_primary=True,

        rrrf_adapter_enabled=False,

        # NEW: Decoder Memory Fusion
        use_corr_decoder_memory=True,

        corr_decoder_memory_channels=80,
    ),
)

load_from = (
    '/workspace/mmdetection3d/data/weights/'
    'ablation_v2/PGFR_Ablation_P1.pth'
)

resume = False

work_dir = (
    'work_dirs/'
    'rrrf_corrbev_dualpath_joint_p1_10k'
)

train_cfg = dict(
    _delete_=True,

    type='IterBasedTrainLoop',

    max_iters=100000,

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

            # CorrBEV Feature Lifter
            'corr_bev_lifter': dict(
                lr_mult=3.0,
            ),

            # Coarse Fusion
            'bbox_head.coarse_bev_adapter': dict(
                lr_mult=3.0,
            ),

            'bbox_head.coarse_bev_gate': dict(
                lr_mult=3.0,
            ),

            'bbox_head.coarse_bev_alpha': dict(
                lr_mult=3.0,
                decay_mult=0.0,
            ),

            # Local Fusion
            'bbox_head.rrrf_query_selector': dict(
                lr_mult=1.0,
            ),

            # NEW: Decoder Memory
            'bbox_head.corr_memory_proj': dict(
                lr_mult=3.0,
            ),

            'bbox_head.corr_memory_gate': dict(
                lr_mult=3.0,
            ),

            'bbox_head.corr_memory_beta': dict(
                lr_mult=5.0,
                decay_mult=0.0,
            ),

            # Transformer Decoder
            'bbox_head.decoder': dict(
                lr_mult=0.3,
            ),
        }
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)

param_scheduler = []

train_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=False,
)

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)

find_unused_parameters = True

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