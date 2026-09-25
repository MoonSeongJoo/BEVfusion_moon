import torch

old = torch.load(
    'data/weights/bevfusion_best_nds70.11_iter_570000_v2.pth',
    map_location='cpu'
)

cur = torch.load(
    'work_dirs/pcc_bevfusion_rrrf_feature_refine_full/iter_123000.pth',
    map_location='cpu'
)

hyb = torch.load(
    'data/weights/hybrid_iros_nds7011_lgpc_iter123000.pth',
    map_location='cpu'
)

old_sd = old['state_dict']
cur_sd = cur['state_dict']
hyb_sd = hyb['state_dict']


# 1. Current LGPC must remain exactly CURRENT
for prefix in [
    'corr.',
    'z_estimator.',
    'calib_head.',
]:
    for k in cur_sd:
        if k.startswith(prefix):
            assert torch.equal(
                cur_sd[k],
                hyb_sd[k]
            ), f'CURRENT LGPC corrupted: {k}'

print('[PASS] Current LGPC preserved')


# 2. Old detector/fusion must be exactly OLD
for prefix in [
    'pts_voxel_encoder.',
    'pts_middle_encoder.',
    'pts_backbone.',
    'pts_neck.',
    'view_transform.',
    'fusion_layer.',
    'feat_projector.',
    'bbox_head.',
]:
    for k in old_sd:

        if not k.startswith(prefix):
            continue

        # Current-only / incompatible keys 제외
        if k not in hyb_sd:
            continue

        if old_sd[k].shape != hyb_sd[k].shape:
            continue

        assert torch.equal(
            old_sd[k],
            hyb_sd[k]
        ), f'OLD transplant failed: {k}'

print('[PASS] Old IROS downstream transplanted')