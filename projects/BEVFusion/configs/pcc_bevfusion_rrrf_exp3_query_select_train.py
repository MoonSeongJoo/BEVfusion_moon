
# ============================================================
# EXP3: Geometry-Selective Attention
#
# Base:
#   RRRF V2 Decoder Tail 20K
#
# Train:
#   Query-selective attention ONLY
#
# Frozen:
#   LGPC, RRRF V2, Fusion,
#   Adapter, Decoder, Prediction Head
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp2_v2_decoder_tail_train.py'
]

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_query_select',

    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
        init_cfg=None,
    ),

    bbox_head=dict(

        rrrf_mode='residual_se3_v2',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,

        query_source='fused',

        # Existing Adapter OFF
        rrrf_adapter_enabled=True,
        rrrf_adapter_scale=0.0,

        # New Query-selective Attention
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,

        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,
    ),
)

# ============================================================
# 2000-iteration pilot
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=999999,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1.0e-4,
        weight_decay=0.01,
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

# ============================================================
# Pretrained 20K Decoder checkpoint
# ============================================================

load_from = (
    'work_dirs/pcc_exp2_v2_decoder_tail_1500/'
    'iter_20000.pth'
)

resume = False
