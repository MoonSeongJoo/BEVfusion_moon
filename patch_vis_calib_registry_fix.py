#!/usr/bin/env python3
"""Patch tools/vis_calib_side_by_side.py (from the earlier ChatGPT draft).

Changes:
  - Register/build PCC BEVFusion only once; instantiate bare LCCNet instead.
  - Read PCC's *existing* GT/Broken extrinsics from Det3DDataSample.
  - Use original LCCNet RGB/depth preprocessing, without Official BEVFusion imports.
  - Apply image/LiDAR augmentation to the LiDAR projection overlay.

The original file is backed up to .bak before editing. No trained weights change.
"""
import ast
from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    sys.exit('Usage: python patch_vis_calib_registry_fix.py tools/vis_calib_side_by_side.py')
p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
if 'build_lccnet_adapter(' in s:
    sys.exit('Already patched: build_lccnet_adapter exists; no changes made.')
required = ['build_runner', 'extract_common_batch_tensors',
            'build_broken_and_clean_calib', 'build_lccnet_corrected_calib',
            'project_points_to_image', 'def main']
for x in required:
    if x not in s:
        sys.exit('Source mismatch: missing '+x+'; original file was not modified')

def replace_func(src, name, replacement):
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name==name]
    if len(funcs)!=1:
        raise ValueError(f'Expected 1 {name} function, got {len(funcs)}')
    lines=src.splitlines(keepends=True)
    n=funcs[0]
    return ''.join(lines[:n.lineno-1])+replacement.rstrip()+'\n\n'+''.join(lines[n.end_lineno:])

s = replace_func(s, 'extract_common_batch_tensors', r'''
def extract_common_batch_tensors(model, raw_batch, device):
    proc = model.data_preprocessor(raw_batch, training=False)
    inp, samples = proc['inputs'], proc['data_samples']
    metas = [s.metainfo for s in samples]
    imgs = inp['imgs']
    if imgs.shape[0] != 1:
        raise RuntimeError('Visualization requires batch_size=1')
    points = get_points_tensor(inp['points'][0])

    # PCC's own dataset provides the very same GT and Broken SE(3)
    # used by its inference path. Never draw a different synthetic error.
    clean = torch.stack([s.camera2lidar for s in samples]).to(imgs.device)
    broken = torch.stack([s.broken_camera2lidar for s in samples]).to(imgs.device)
    intr = torch.stack([s.broken_camera_intrinsics for s in samples]).to(imgs.device)

    def aug(name, per_camera):
        result=[]
        for meta in metas:
            arr=np.asarray(meta.get(name, np.eye(4)), dtype=np.float32)
            if per_camera and arr.ndim==2:
                arr=np.broadcast_to(arr, (imgs.shape[1],4,4)).copy()
            if not per_camera and arr.ndim==3:
                arr=arr[0]
            result.append(arr)
        return imgs.new_tensor(np.stack(result,axis=0))

    return dict(batch_inputs=inp, batch_data_samples=samples,
                batch_input_metas=metas, imgs=imgs, points=points,
                camera2lidar=clean, broken_camera2lidar=broken,
                cam2img=intr, img_aug_matrix=aug('img_aug_matrix',True),
                lidar_aug_matrix=aug('lidar_aug_matrix',False))
''')

s = replace_func(s, 'build_broken_and_clean_calib', r'''
def build_broken_and_clean_calib(ours_model, data_dict):
    # In PCC the perturbation is materialized by its dataset and stored
    # in data_samples; _build_broken_camera2lidar belongs to Official only.
    broken = data_dict['broken_camera2lidar']
    clean = data_dict['camera2lidar']
    K = data_dict['cam2img']
    return dict(broken_camera2lidar=broken,
                broken_calib=ours_model._build_calib_dict_from_cam2lidar(broken,K),
                clean_calib=ours_model._build_calib_dict_from_cam2lidar(clean,K))
''')

s = replace_func(s, 'build_lccnet_corrected_calib', r'''
def build_lccnet_corrected_calib(lcc_wrapper, data_dict, broken_camera2lidar):
    from projects.LCCNetModern.lie import se3
    from projects.LCCNetModern.calib_geometry import project_lidar_to_sparse_depth
    imgs=data_dict['imgs']
    B,N,C,H,W=imgs.shape
    if (H,W)!=(256,704):
        raise RuntimeError(f'LCCNet checkpoint expects image 256x704, got {(H,W)}')
    K=data_dict['cam2img']
    broken_calib=lcc_wrapper._build_calib_dict_from_cam2lidar(broken_camera2lidar,K)

    # This function expects a LIST of point clouds, not one [P,5] tensor.
    depth=project_lidar_to_sparse_depth(
        points=[data_dict['points']],
        lidar2image=broken_calib['lidar2img'],
        img_aug_matrix=data_dict['img_aug_matrix'],
        lidar_aug_matrix=data_dict['lidar_aug_matrix'],
        image_hw=(H,W)) / lcc_wrapper.lccnet_depth_scale
    rgb=lcc_wrapper._bevfusion_img_to_lccnet(imgs)
    rgb=rgb.reshape(B*N,3,H,W)
    depth=depth.reshape(B*N,1,H,W)
    outs=[]
    with torch.inference_mode():
        for start in range(0,B*N,lcc_wrapper.lccnet_cam_chunk):
            end=min(start+lcc_wrapper.lccnet_cam_chunk,B*N)
            outs.append(lcc_wrapper.lccnet(rgb[start:end].float(),depth[start:end].float()))
    pred_x=torch.cat(outs,dim=0)
    assert pred_x.shape==(B*N,6) and torch.isfinite(pred_x).all()
    correction_pred=se3.exp(pred_x).reshape(B,N,4,4)
    corrected=correction_pred @ broken_camera2lidar
    return lcc_wrapper._build_calib_dict_from_cam2lidar(corrected,K)
''')

s = replace_func(s, 'project_points_to_image', r'''
def project_points_to_image(points_lidar_frame, lidar2img_4x4, H, W,
                            img_aug=None, lidar_aug=None):
    # Match projection convention of LCCNetModern/calib_geometry.py.
    pts=points_lidar_frame[:,:3].to(dtype=torch.float32)
    P=lidar2img_4x4.to(device=pts.device,dtype=pts.dtype)
    if lidar_aug is not None:
        A=lidar_aug.to(device=pts.device,dtype=pts.dtype)
        pts=(torch.linalg.inv(A[:3,:3]) @ (pts-A[:3,3]).T).T
    xyz1=torch.cat((pts,torch.ones_like(pts[:,:1])),dim=1)
    proj=(P[:3,:]@xyz1.T).T
    z=proj[:,2].clone()
    good=torch.isfinite(proj).all(dim=1)&(z>1e-5)
    zsafe=z.clamp(min=1e-5)
    geom=torch.stack((proj[:,0]/zsafe,proj[:,1]/zsafe,z),dim=1)
    if img_aug is not None:
        A=img_aug.to(device=pts.device,dtype=pts.dtype)
        geom=(A[:3,:3]@geom.T).T+A[:3,3]
    uv=geom[:,:2]
    good &= torch.isfinite(uv).all(dim=1)
    good &= (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    return torch.round(uv[good]).int().cpu().numpy()
''')

# Drop obsolete import-only lines. The name 'OfficialBEVFusion' must not be imported.
s=s.replace('    lcc_runner = build_runner(args.lccnet_config, args.lccnet_checkpoint, args.device)\n',
'''    lcc_model = build_lccnet_adapter(
        ours_model, args.lccnet_model_checkpoint, args.device)
''')
s=s.replace('    lcc_model = lcc_runner.model\n', '')
# ensure valid main invocation
if 'lcc_runner = build_runner' in s:
    sys.exit('Unexpected lcc_runner line remained; file not modified')

# We intentionally require the actual nuScenes-trained LCCNet model .pth,
# not the Official BEVFusion detector checkpoint.
needle='    return parser.parse_args()'
if needle not in s:
    sys.exit('parse_args ending not found; no file modified')
s=s.replace(needle, '''    parser.add_argument('--lccnet-model-checkpoint', required=True,
                        help='NUScenes-trained LCCNet best_joint_model.pth')
'''+needle,1)
for option in ('--lccnet-config','--lccnet-checkpoint'):
    # Keep deprecated flags accepted for copy-paste compatibility, but optional.
    pat=re.compile(r'(parser\.add_argument\(\s*[\'\"]'+re.escape(option)+r'[\'\"].*?\))',re.S)
    s,n=pat.subn(lambda m:m.group(1).replace('required=True','required=False'),s,count=1)
    if n!=1:
        sys.exit('Failed to locate argument '+option+'; no file modified')

# Build pretrained LCCNet directly: exact network/metadata checks as Official
# _init_lccnet_baseline, but WITHOUT its conflicting @MODELS registration.
insert=r'''
def build_lccnet_adapter(ours_model, ckpt_path, device):
    from projects.LCCNetModern.lccnet import LCCNet
    ckpt=torch.load(ckpt_path, map_location='cpu', weights_only=False)
    meta=ckpt.get('args',{})
    if ckpt.get('resolution') != [256,704]:
        raise RuntimeError('LCCNet checkpoint resolution must be [256,704]')
    if int(meta.get('use_feat_from',-1)) != 1:
        raise RuntimeError('LCCNet requires use_feat_from=1')
    if meta.get('input_color_order') != 'rgb':
        raise RuntimeError('LCCNet checkpoint must use RGB input')
    depth_scale=float(meta.get('depth_scale',-1))
    if abs(depth_scale-50.)>1e-6:
        raise RuntimeError(f'Unexpected depth_scale: {depth_scale}')

    net=LCCNet(resnet_argv=dict(num_layers=18,pretrained=False,frozen=False),
               image_size=(256,704),use_feat_from=1,md=4,
               use_reflectance=False,dropout=0.0,
               Action_Func='leakyrelu',attention=False)
    net.load_state_dict(ckpt['model'],strict=True)
    net.to(device).eval().requires_grad_(False)

    class Adapter:
        pass
    a=Adapter()
    a.lccnet=net
    a.lccnet_depth_scale=depth_scale
    a.lccnet_cam_chunk=6
    a._build_calib_dict_from_cam2lidar=ours_model._build_calib_dict_from_cam2lidar

    def normalize_rgb(imgs):
        # Match OfficialBEVFusion._bevfusion_img_to_lccnet EXACTLY.
        x=imgs.float()
        if x.min().item()>=0. and x.max().item()>2.:
            x=x/255.
        mean=x.new_tensor([.485,.456,.406]).view(1,1,3,1,1)
        std=x.new_tensor([.229,.224,.225]).view(1,1,3,1,1)
        return (x-mean)/std
    a._bevfusion_img_to_lccnet=normalize_rgb
    print('[OK] Direct LCCNet checkpoint loaded:',ckpt_path)
    print('[OK] epoch:',ckpt.get('epoch','unknown'),'depth_scale:',depth_scale)
    return a
'''
# insert preceding the script's main def
s=s.replace('def main():',insert+'\n\ndef main():',1)

# Augmentation-aware overlays. The original script drew *raw* projection on
# a 256x704 augmented image, leading to mismatched pixels.
rgx=re.compile(r'(project_points_to_image\(\s*points_for_vis,.*?\bH\s*,\s*W\s*)(\))',re.S)
counts=[0]
def with_aug(m):
    counts[0]+=1
    return (m.group(1)+',\n            img_aug=data_dict[\'img_aug_matrix\'][0, cam_idx],\n'
            '            lidar_aug=data_dict[\'lidar_aug_matrix\'][0]\n        '+m.group(2))
s=rgx.sub(with_aug,s)
if counts[0]!=5:
    sys.exit(f'Expected five display projection calls, found {counts[0]}; no file modified')

ast.parse(s)
backup=p.with_suffix(p.suffix+'.bak')
if backup.exists():
    sys.exit(f'Backup already exists: {backup}. Do not overwrite; no changes made')
backup.write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
p.write_text(s, encoding='utf-8')
print('[PATCHED]',p)
print('[BACKUP]',backup)
print('Important: --lccnet-model-checkpoint must be best_joint_model.pth (not official detector .pth).')
