# Stage 2: fine-tune a converged Ver4.0 Dual-Path RRRF + Transformer Decoder
# from the 20K best NDS checkpoint. This is a NEW 0..6000 iteration run.
# Place alongside rrrf_corrbev_dualpath_joint_p1_nds_earlystop.py.
_base_ = ['./rrrf_corrbev_dualpath_joint_p1_train.py']

# Important: verify this checkpoint exists; best checkpoint can be removed
# automatically if a later validation improves NDS.
load_from = (
    '/workspace/mmdetection3d/work_dirs/'
    'rrrf_corrbev_dualpath_joint_p1_10k/'
    'best_NDS_iter_20000.pth'
)
resume = False  # load weights only; restart optimizer and scheduler
work_dir = '/workspace/mmdetection3d/work_dirs/rrrf_dualpath_phase2_best20k_6k'

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=6000,
    val_interval=2000,
)

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
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2),
)

# Iteration-based warmup + cosine decay. eta_min_ratio applies to each
# parameter group's own initial LR; relative LR multipliers are retained.
param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=False,
        start_factor=0.5,
        begin=0,
        end=200,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        T_max=5800,
        eta_min_ratio=0.1,
        begin=200,
        end=6000,
    ),
]

# Inherits: val_evaluator, val_dataloader, save_best='NDS',
# StopOnNDSThresholdHook(threshold=0.7069), and model joint-freeze stage.
