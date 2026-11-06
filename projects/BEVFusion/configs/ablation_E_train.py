# ablation_E_train.py

_base_ = ['./ablation_D_train.py']

model = dict(
    bbox_head=dict(
        rrrf_use_coarse=True,
        rrrf_use_refined=True,

        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,

        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,
    ),
)

work_dir = 'work_dirs/ablation_v2_E'