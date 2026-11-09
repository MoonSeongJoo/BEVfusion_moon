# pgfr_coarse_dual_e1_eval.py

# Inherit model, dataset and test settings
_base_ = [
    './pgfr_coarse_dual_e1_train.py'
]

# ==============================================
# 1. Disable training configuration
# ==============================================

train_dataloader = None
train_cfg = None
optim_wrapper = None

param_scheduler = None
train_evaluator = None

# ==============================================
# 2. Disable validation configuration
# ==============================================

val_dataloader = None
val_cfg = None
val_evaluator = None

# ==============================================
# 3. Checkpoint is passed via dist_test.sh
# ==============================================

load_from = None
resume = False

# ==============================================
# 4. Keep inherited test settings
# ==============================================
# test_cfg
# test_dataloader
# test_evaluator
#
# Keep inherited Dual-Source model configuration
# including rrrf_coarse_dual_enabled=True.