_base_ = [
    './pgfr_feature_refine_local_train.py'
]

# 5000 iteration training
train_cfg = dict(
    max_iters=5000,
    val_interval=999999
)

# Disable validation during training
val_cfg = None
val_dataloader = None
val_evaluator = None

# Reduce CPU RAM usage and avoid DataLoader worker failures
train_dataloader = dict(
    num_workers=0,
    persistent_workers=False
)

# Logging and checkpoints
default_hooks = dict(
    logger=dict(
        interval=20
    ),
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=10,
        save_last=True
    )
)

# Start a fresh run from the pretrained checkpoint
# inherited from the parent config.
resume = False

work_dir = 'work_dirs/pgfr_feature_refine_local_5k'
