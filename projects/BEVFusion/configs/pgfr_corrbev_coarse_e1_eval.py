_base_ = [
    './pgfr_corrbev_coarse_e1_train.py'
]

train_cfg = None
train_dataloader = None
optim_wrapper = None
param_scheduler = None

val_cfg = None
val_dataloader = None
val_evaluator = None

load_from = None
resume = False