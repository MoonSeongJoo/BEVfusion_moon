# CorrNet Aligned BEV + Coarse Dual-Source

_base_ = [
    './pgfr_coarse_dual_e1_train.py'
]

model = dict(
    use_corr_aligned_bev=True,
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'corr_bev_lifter': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_adapter': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_gate': dict(
                lr_mult=5.0,
            ),

            'bbox_head.coarse_bev_alpha': dict(
                lr_mult=5.0,
                decay_mult=0.0,
            ),
        }
    )
)

# Start from the same P1 5K checkpoint
# for a controlled comparison.

load_from = (
    '/workspace/mmdetection3d/work_dirs/'
    'p1_local_gate_target_5k/iter_5000.pth'
)

resume = False

work_dir = 'work_dirs/p1_corrbev_coarse_2k'