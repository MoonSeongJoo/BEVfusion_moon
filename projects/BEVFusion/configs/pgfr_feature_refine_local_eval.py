# PGFR-Det: Feature Refine 5K Evaluation

_base_ = [
    './pgfr_feature_refine_local_5k.py'
]

# Independent test loop
test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

# Lower CPU memory usage for Kubernetes
test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
)

# Inherited from training config:
# calibration_mode = 'pcc_full'
# rrrf_mode = 'feature_refine'
# rrrf_query_select_enabled = True
# rrrf_adapter_enabled = False
# test_evaluator includes NuScenesMetric

load_from = None
resume = False

work_dir = 'work_dirs/pgfr_feature_refine_local_5k_eval'

default_hooks = dict(
    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),
)