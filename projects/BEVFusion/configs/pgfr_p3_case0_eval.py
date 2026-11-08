_base_ = ['./pgfr_feature_refine_local_eval.py']

model = dict(
    enable_selective_freezing=False,
    lgpc_train_stage='rrrf_ablation',

    bbox_head=dict(
        rrrf_mode='feature_refine',
        query_source='fused',
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
