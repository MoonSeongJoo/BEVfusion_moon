_base_ = [
    './pcc_bevfusion_repair_v1_corr_r1_ft3k.py'
]


model = dict(

    enable_selective_freezing=False,

    lgpc_train_stage='corr_refine',


    bbox_head=dict(
        rrrf_mode='lgpc_only',
    ),


    corr=dict(

        init_cfg=None,

        enable_cycle=False,

        enable_local_refine=True,

        local_refine_in_channels=512,

        local_refine_proj_dim=64,

        local_refine_hidden_dim=128,

        local_refine_radius_px=64.0,

        local_refine_grid_size=5,

        local_refine_temperature=1.0,

        # ====================================================
        # R2.1
        #
        # normalized candidate CE ~= 1 at initialization.
        #
        # weighted contribution ~= 0.01
        # ====================================================

        local_refine_candidate_loss_weight=0.01,
    ),
)


# ★ NEW BASE
load_from = (
    'data/work_dirs/'
    'pcc_repair_v1_lgpc_full24_ddp2/'
    'epoch_14.pth'
)

resume = False


train_cfg = dict(

    _delete_=True,

    type='IterBasedTrainLoop',

    max_iters=500,

    val_interval=1000000,
)


optim_wrapper = dict(

    optimizer=dict(
        lr=1.0e-4,
    ),
)


param_scheduler = []


default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(

        _delete_=True,

        type='CheckpointHook',

        by_epoch=False,

        interval=250,

        max_keep_ckpts=1,

        save_last=True,
    ),
)