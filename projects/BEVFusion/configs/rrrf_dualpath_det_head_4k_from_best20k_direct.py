# PGFR-Det Ver4.0: Phase-3 RRRF + Decoder + Final 3D Prediction Head
# Starts from trained Dual-Path Best20K weights (not the original P1 model).
# Inherits the known-good P1 dual-path experiment directly, without an
# unnecessary intermediate Phase-2 config dependency.
_base_ = ['./rrrf_corrbev_dualpath_joint_p1_train.py']

model = dict(
    lgpc_train_stage='rrrf_dualpath_det_head',
    enable_selective_freezing=False,
)

# First copy the original best checkpoint to this separate backup path.
load_from = (
    '/workspace/mmdetection3d/work_dirs/rrrf_dualpath_det_head_4k/iter_1000.pth'
)
resume = False
work_dir = '/workspace/mmdetection3d/work_dirs/rrrf_dualpath_det_head_4k_1'

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=5000,
)

# Existing val_dataloader/val_evaluator and NDS threshold hook inherited.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-6, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'corr_bev_lifter': dict(lr_mult=2.0),
            'bbox_head.coarse_bev_adapter': dict(lr_mult=2.0),
            'bbox_head.coarse_bev_gate': dict(lr_mult=2.0),
            'bbox_head.coarse_bev_alpha': dict(lr_mult=2.0, decay_mult=0.0),
            'bbox_head.rrrf_query_selector': dict(lr_mult=1.0),
            'bbox_head.corr_memory_proj': dict(lr_mult=2.0),
            'bbox_head.corr_memory_gate': dict(lr_mult=2.0),
            'bbox_head.corr_memory_beta': dict(lr_mult=3.0, decay_mult=0.0),
            'bbox_head.decoder': dict(lr_mult=0.2),
            'bbox_head.prediction_heads.0': dict(lr_mult=0.4),
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', by_epoch=False, start_factor=0.5,
        begin=0, end=200,
    ),
    dict(
        type='CosineAnnealingLR', by_epoch=False,
        T_max=3800, eta_min_ratio=0.1,
        begin=200, end=4000,
    ),
]

# Explicit best-checkpoint policy for Phase-3 (including early-stop order).
# The inherited StopOnNDSThresholdHook must be registered on the server.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=5,
        save_last=True,
        save_best='NDS',
        rule='greater',
    ),
)
