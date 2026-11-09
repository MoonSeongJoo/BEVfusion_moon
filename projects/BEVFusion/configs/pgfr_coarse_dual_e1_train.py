# E1: Coarse Dual-Source
# Coarse: CorrNet + Corrected Camera BEV
# Refined: CorrNet only
# Local: OFF

_base_ = [
    './pgfr_feature_refine_local_5k.py'
]

model = dict(
    calibration_mode='pcc_full',

    enable_selective_freezing=False,

    lgpc_train_stage='rrrf_joint_fresh',

    bbox_head=dict(
        rrrf_mode='feature_refine',

        query_source='fused',

        rrrf_use_coarse=True,
        rrrf_use_refined=True,

        # Dual-Source ON
        rrrf_coarse_dual_enabled=True,
        rrrf_coarse_dual_bev_channels=80,
        rrrf_coarse_dual_scale=1.0,

        # Local OFF
        rrrf_query_select_enabled=False,
        rrrf_query_select_scale=0.0,

        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=999999,
)

val_cfg = None
val_dataloader = None
val_evaluator = None

train_dataloader = dict(
    num_workers=12,
    persistent_workers=False,
)

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
            'bbox_head.coarse_bev_adapter': dict(
                lr_mult=5.0
            ),
            'bbox_head.coarse_bev_gate': dict(
                lr_mult=5.0
            ),
            'bbox_head.coarse_bev_alpha': dict(
                lr_mult=5.0,
                decay_mult=0.0,
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
        max_keep_ckpts=8,
        save_last=True,
    ),
)

load_from = (
    '/workspace/mmdetection3d/work_dirs/'
    'p1_local_gate_target_5k/iter_5000.pth'
)

resume = False

work_dir = 'work_dirs/p1_coarse_dual_e1_2k'