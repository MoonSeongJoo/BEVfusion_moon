# ablation_C_train.py

_base_ = ['./ablation_B_train.py']

model = dict(
    bbox_head=dict(
        rrrf_use_coarse=True,
        rrrf_use_refined=False,
        rrrf_query_select_enabled=False,
        rrrf_query_select_scale=0.0,
    ),
)

work_dir = 'work_dirs/ablation_v2_C'