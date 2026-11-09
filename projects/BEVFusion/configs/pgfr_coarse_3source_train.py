# Coarse 3-source input: CorrNet 2D + Camera-only BEV + CorrNet BEV.
# Initialize from original P1 5K (NOT old E1 80ch checkpoint).
_base_ = ['./pgfr_corrbev_coarse_e1_train.py']

model = dict(
    use_corr_aligned_bev=True,
    corr_bev_feature_scale=1.0,
    bbox_head=dict(
        rrrf_coarse_dual_enabled=True,
        rrrf_coarse_dual_bev_channels=160,
        rrrf_coarse_dual_scale=1.0,
    ),
)

# NOTE: inherited load_from must point to 80ch-free P1 (no E1 adapter).
load_from = (
    '/workspace/mmdetection3d/work_dirs/'
    'p1_local_gate_target_5k/iter_5000.pth'
)
resume = False
work_dir = 'work_dirs/p1_coarse_3source_2k'
