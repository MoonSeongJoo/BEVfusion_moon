
# ============================================================
# Ablation A: Common P0 Baseline
# No RRRF, No Fine-tuning
# ============================================================

_base_ = ['./ablation_B_train.py']

model = dict(
    enable_selective_freezing=False,
    calibration_mode='pcc_full',
    lgpc_train_stage='rrrf_ablation',

    bbox_head=dict(
        rrrf_mode='feature_refine',
        query_source='fused',
        rrrf_detach_geometry=True,

        rrrf_use_coarse=False,
        rrrf_use_refined=False,

        rrrf_query_select_enabled=False,
        rrrf_query_select_scale=0.0,

        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,

        rrrf_ablation_identity_geometry=False,
        rrrf_ablation_skip_refined_attention=False,
    ),
)

load_from = None
resume = False

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

work_dir = 'work_dirs/ablation_v2_A_eval'
