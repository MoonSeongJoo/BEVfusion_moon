_base_ = [
    './pcc_bevfusion_repair_v1_corr_r1_ft3k.py'
]


# ============================================================
# R2: Local Corr Refinement Pilot
# ============================================================

model = dict(

    enable_selective_freezing=False,

    lgpc_train_stage='corr_refine',


    bbox_head=dict(
        rrrf_mode='lgpc_only',
    ),


    corr=dict(

        init_cfg=None,

        enable_cycle=False,

        # ====================================================
        # NEW
        # ====================================================

        enable_local_refine=True,

        local_refine_in_channels=512,

        local_refine_proj_dim=64,

        local_refine_hidden_dim=128,

        local_refine_radius_px=64.0,

        local_refine_grid_size=5,

        local_refine_temperature=1.0,
    ),
)


# ============================================================
# R1 iter3000 = fixed coarse Corr provider
#
# IMPORTANT:
#
# resume=False
#
# New LocalRefiner does not exist in this checkpoint.
# ============================================================

load_from = (
    'data/work_dirs/'
    'pcc_corr_r1_unique_pixelbalanced_ft3k/'
    'iter_3000.pth'
)

resume = False


# ============================================================
# 500-iteration pilot
# ============================================================

train_cfg = dict(
    _delete_=True,

    type='IterBasedTrainLoop',

    max_iters=500,

    val_interval=1000000,
)


# ============================================================
# Only the new head is trainable.
#
# It starts from scratch, therefore 1e-4 is reasonable.
# ============================================================

optim_wrapper = dict(

    optimizer=dict(
        lr=1.0e-4,
    ),
)


# No scheduler during short diagnostic.
param_scheduler = []


# ============================================================
# Logging / checkpoint
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=10,
    ),

    checkpoint=dict(
        _delete_=True,

        type='CheckpointHook',

        by_epoch=False,

        interval=250,

        max_keep_ckpts=2,

        save_last=True,
    ),

    visualization=dict(
        _delete_=True,

        type='Det3DVisualizationHook',

        draw=False,
    ),
)