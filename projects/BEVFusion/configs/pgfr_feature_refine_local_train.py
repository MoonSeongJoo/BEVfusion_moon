
# pgfr_feature_refine_local_train.py

_base_ = [
    './pcc_bevfusion_rrrf_exp4_joint_fresh_train.py'
]

model = dict(
    enable_selective_freezing=False,
    calibration_mode='pcc_full',
    lgpc_train_stage='rrrf_joint_fresh',

    bbox_head=dict(
        # No Stage-2 residual SE(3) prediction
        rrrf_mode='feature_refine',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,

        # Original fused BEV queries
        query_source='fused',

        # Disable feature adapter in first experiment
        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,

        # Enable GeometrySelectiveAttention
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,
        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,

        # No ablations
        rrrf_ablation_identity_geometry=False,
        rrrf_ablation_skip_refined_attention=False,
    ),
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,

    # Validate separately after pilot training
    val_interval=999999,
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
            'bbox_head.rrrf_query_selector': dict(lr_mult=2.0),
            'bbox_head.decoder.0': dict(lr_mult=1.0),
            'bbox_head.prediction_heads.0': dict(lr_mult=1.0),
        },
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

# Must be changed to a verified checkpoint.
load_from = (
    '/workspace/mmdetection3d/data/weights/PGFR_Det_final_model.pth'
)

# Load model weights only, start a new training run.
resume = False

work_dir = 'work_dirs/pgfr_feature_refine_local_2k'
