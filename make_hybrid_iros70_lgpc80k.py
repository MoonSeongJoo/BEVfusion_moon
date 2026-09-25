import copy
import torch


OLD_CKPT = (
    'data/weights/'
    'bevfusion_best_nds70.11_iter_570000_v2.pth'
)

CURRENT_CKPT = (
    'work_dirs/'
    'pcc_bevfusion_rrrf_feature_refine_full/'
    'iter_123000.pth'
)

OUTPUT_CKPT = (
    'data/weights/'
    'hybrid_iros_nds7011_lgpc_iter123000.pth'
)


def get_state_dict(ckpt):
    if 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def canonical_key(k):
    if k.startswith('module.'):
        return k[len('module.'):]
    return k


old_ckpt = torch.load(
    OLD_CKPT,
    map_location='cpu',
)

cur_ckpt = torch.load(
    CURRENT_CKPT,
    map_location='cpu',
)

old_sd = get_state_dict(old_ckpt)
cur_sd = get_state_dict(cur_ckpt)


# ------------------------------------------------------------
# Current LGPC must remain untouched.
#
# Only transplant the old high-NDS downstream network.
# ------------------------------------------------------------

copy_prefixes = (
    'pts_voxel_encoder.',
    'pts_middle_encoder.',
    'pts_backbone.',
    'pts_neck.',

    'view_transform.',
    'fusion_layer.',

    'feat_projector.',

    'bbox_head.',
)


# Canonical old-key map
old_map = {
    canonical_key(k): (k, v)
    for k, v in old_sd.items()
}


merged_sd = copy.deepcopy(cur_sd)

copied = []
missing = []
shape_mismatch = []


for cur_key, cur_tensor in cur_sd.items():

    ckey = canonical_key(cur_key)

    if not ckey.startswith(copy_prefixes):
        continue

    if ckey not in old_map:
        missing.append(ckey)
        continue

    old_key, old_tensor = old_map[ckey]

    if tuple(old_tensor.shape) != tuple(cur_tensor.shape):

        shape_mismatch.append(
            (
                ckey,
                tuple(old_tensor.shape),
                tuple(cur_tensor.shape),
            )
        )

        continue

    merged_sd[cur_key] = old_tensor.clone()

    copied.append(ckey)


# ------------------------------------------------------------
# Make sure CURRENT LGPC weights have not been replaced.
# ------------------------------------------------------------

protected_prefixes = (
    'corr.',
    'z_estimator.',
    'calib_head.',
    'img_backbone.',
    'img_neck.',
    'img_bbox_head.',
)


for key, original_tensor in cur_sd.items():

    ckey = canonical_key(key)

    if ckey.startswith(protected_prefixes):

        if not torch.equal(
            original_tensor,
            merged_sd[key],
        ):

            raise RuntimeError(
                f'Protected CURRENT weight was modified: {ckey}'
            )


# ------------------------------------------------------------
# Save using CURRENT checkpoint as container/meta.
# ------------------------------------------------------------

out_ckpt = copy.deepcopy(cur_ckpt)

if 'state_dict' in out_ckpt:
    out_ckpt['state_dict'] = merged_sd
else:
    out_ckpt = merged_sd


torch.save(
    out_ckpt,
    OUTPUT_CKPT,
)


print('=========================================')
print('HYBRID CHECKPOINT CREATED')
print('=========================================')
print(f'current base : {CURRENT_CKPT}')
print(f'old donor    : {OLD_CKPT}')
print(f'output       : {OUTPUT_CKPT}')
print()
print(f'copied       : {len(copied)}')
print(f'missing      : {len(missing)}')
print(f'shape mismatch: {len(shape_mismatch)}')

print('\n--- copied samples ---')
for k in copied[:40]:
    print(k)

print('\n--- shape mismatch ---')
for x in shape_mismatch[:40]:
    print(x)

print('=========================================')