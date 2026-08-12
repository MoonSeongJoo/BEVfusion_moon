import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch.nn.functional as F
import torchvision

def display_depth_maps(imgs, depth_map, mis_calibrated_depth_map):
    # """
    # 이미지 위에 depth map과 mis_calibrated depth map을 오버레이하여 디스플레이하는 함수
    # """

    for cid in range(imgs.shape[0]):

        img_np = imgs.squeeze()[cid].cpu().detach().numpy()
        depth_gt_np = depth_map.squeeze()[cid].cpu().detach().numpy() 
        depth_mis_np = mis_calibrated_depth_map.squeeze()[cid].cpu().detach().numpy()
        # sbs_img_np = sbs_img.squeeze()[cid].cpu().detach().numpy()
        img_np = np.transpose(img_np,(1,2,0))
        depth_gt_np = np.transpose(depth_gt_np,(1,2,0))
        depth_mis_np = np.transpose(depth_mis_np,(1,2,0))
        # sbs_img_np = np.transpose(sbs_img_np,(1,2,0))
        # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            depth_gt_np = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min())
            depth_mis_np = (depth_mis_np - depth_mis_np.min()) / (depth_mis_np.max() - depth_mis_np.min())
            # sbs_img_np = (sbs_img_np - sbs_img_np.min()) / (sbs_img_np.max() - sbs_img_np.min())

        # input display
        ####### display input signal #########        
        plt.figure(figsize=(20, 20))
        plt.subplot(311)
        plt.imshow(img_np)
        plt.title("camera_input", fontsize=15)
        plt.axis('off')

        plt.subplot(312)
        plt.imshow( depth_gt_np, cmap='magma')
        plt.title("calibrated_lidar_input", fontsize=15)
        plt.axis('off') 

        plt.subplot(313)
        plt.imshow( depth_mis_np, cmap='magma')
        plt.title("mis-calibrated_lidar_input", fontsize=15)
        plt.axis('off')

        # plt.subplot(224)
        # plt.imshow( sbs_img_np)
        # plt.title("sbs_img_input", fontsize=15)
        # plt.axis('off')
        
        plt.tight_layout(pad=0)  # 여백 제거
        plt.savefig(f'raw_input_image_{cid+1}.png', dpi=300, bbox_inches='tight')
        plt.close('all')
    
    print ('display end')
    # ############ end of display input signal ###################

def visualize_depth_maps(lidar_depth_map_gt, lidar_depth_map_mis):
    num_cameras = lidar_depth_map_gt.shape[0]
    fig, axes = plt.subplots(num_cameras, 2, figsize=(12, 6*num_cameras))
    
    for i in range(num_cameras):
        # Ground Truth
        gt = lidar_depth_map_gt[i].cpu().numpy() if torch.is_tensor(lidar_depth_map_gt) else lidar_depth_map_gt[i]
        axes[i,0].imshow(gt, cmap='viridis', vmin=0, vmax=80)  # 거리 범위 0-80m 가정
        
        # Miscalibrated
        mis = lidar_depth_map_mis[i]
        if mis.ndim == 3:  # (C,H,W) → (H,W,C)
            mis = np.transpose(mis, (1,2,0))
        mis = (mis - np.min(mis)) / (np.max(mis) - np.min(mis) + 1e-8)
        axes[i,1].imshow(mis, vmin=0, vmax=1)
        
    plt.savefig('depth_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()
#     print('oeverlay imgs lidar shape' , lidar.shape)

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
#     print('blended_img shape' , np.asarray(blended_img).shape)
    # io.imshow(blended_img)
    # io.show()
    # plt.figure()
    # plt.imshow(lidar_color)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img , rgb , lidar_color

def points2depthmap_grid(points, height, width):
    downsample =1
    grid_config = {
                    'x': [-51.2, 51.2, 0.8],
                    'y': [-51.2, 51.2, 0.8],
                    'z': [-5, 3, 8],
                    'depth': [1.0, 60.0, 0.5], # original
                    # 'depth': [1.0, 80.0, 0.5],
                }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), device=points.device, dtype=torch.float32)
    depth_map_grid = torch.zeros((height, width ,2), device=points.device, dtype=torch.float32)
    coor = torch.round(points[:, :2] / downsample) # uv
    depth = points[:, 2] # z
    # kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
    #     coor[:, 1] >= 0) & (coor[:, 1] < height) & (
    #         depth < grid_config['depth'][1]) & (
    #             depth >= grid_config['depth'][0])
    # coor, depth = coor[kept1], depth[kept1]
    # ranks = coor[:, 0] + coor[:, 1] * width
    # sort = (ranks + depth / 100.).argsort()
    # coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    # kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    # kept2[1:] = (ranks[1:] != ranks[:-1])
    # coor_float, depth = coor[kept2], depth[kept2]
    # coor_long = coor.to(torch.long)
    # depth_map[coor_long[:, 1], coor_long[:, 0]] = depth
    # depth_map_grid[coor_long[:, 1], coor_long[:, 0], :] = coor
    return coor, depth

import numpy as np

def points2depthmap_cpu(points, height, width, downsample=1):
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = np.zeros((height, width), dtype=np.float32)
    coor = np.round(points[:, :2] / downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = np.argsort(ranks + depth / 100.)
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = np.ones(coor.shape[0], dtype=bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.astype(np.int64)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # 최종 유효 인덱스 계산
    valid_indices = np.where(kept1)[0][kept2]
    
    return depth_map, coor, depth, valid_indices

# def points2depthmap_gpu(points, height, width, grid_config, downsample=1):
#     device = points.device
#     # grid_config = {
#     #     'x': [-51.2, 51.2, 0.8],
#     #     'y': [-51.2, 51.2, 0.8],
#     #     'z': [-5, 3, 8],
#     #     # 'depth': [1.0, 60.0, 0.5],
#     #     'depth': [0.0, 60.0, 0.5],
#     # }
#     height, width = height // downsample, width // downsample
#     depth_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    
#     # 모든 연산을 GPU에서 수행하도록 명시적으로 device 지정
#     coor = torch.round(points[:, :2] / downsample).to(device)
#     depth = points[:, 2].to(device)
    
#     # 벡터화된 연산으로 kept1 계산
#     kept1 = ((coor[:, 0] >= 0) & (coor[:, 0] < width) &
#              (coor[:, 1] >= 0) & (coor[:, 1] < height) &
#              (depth < grid_config['depth'][1]) &
#              (depth >= grid_config['depth'][0]))
    
#     coor, depth = coor[kept1], depth[kept1]
    
#     # width를 텐서로 변환하여 GPU에서 연산
#     width_tensor = torch.tensor(width, device=device)
#     ranks = coor[:, 0] + coor[:, 1] * width_tensor
    
#     # argsort 연산은 이미 GPU에서 수행됨
#     # sort = (ranks + depth / 100.).argsort()
#     # 수정 코드 (Z값 정상화)
#     # sort = (ranks - depth / 100.).argsort(descending=True)  # 내림차순 정렬
#     sort = (ranks - depth).argsort(descending=True)
    
#     coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

#     # 벡터화된 연산으로 kept2 계산
#     kept2 = torch.ones(coor.shape[0], device=device, dtype=torch.bool)
#     kept2[1:] = (ranks[1:] != ranks[:-1])
    
#     coor, depth = coor[kept2], depth[kept2]
    
#     coor = coor.to(torch.long)
    
#     # GPU에서 인덱싱 연산 수행
#     depth_map[coor[:, 1], coor[:, 0]] = depth

#     # # 최종 유효 인덱스 계산
#     valid_indices = torch.where(kept1)[0][kept2]
    
#     return depth_map ,coor, depth, valid_indices

def points2depthmap_gpu(points, height, width, grid_config, downsample=1):
    device = points.device
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    
    # 원본 포인트 인덱스 초기화 (0부터 순차적)
    original_indices = torch.arange(points.size(0), device=device)
    
    coor = torch.round(points[:, :2] / downsample).to(device)
    depth = points[:, 2].to(device)
    
    # 1차 필터링 (좌표 범위 및 깊이 조건)
    kept1 = ((coor[:, 0] >= 0) & (coor[:, 0] < width) &
             (coor[:, 1] >= 0) & (coor[:, 1] < height) &
             (depth < grid_config['depth'][1]) &
             (depth >= grid_config['depth'][0]))
    
    coor, depth = coor[kept1], depth[kept1]
    original_indices = original_indices[kept1]  # 인덱스 추적
    
    # 랭크 계산 및 정렬
    width_tensor = torch.tensor(width, device=device)
    ranks = coor[:, 0] + coor[:, 1] * width_tensor
    sort = (ranks - depth).argsort(descending=True)
    
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
    original_indices = original_indices[sort]  # 정렬된 인덱스
    
    # 2차 필터링 (중복 좌표 제거)
    kept2 = torch.ones(coor.shape[0], device=device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    
    coor, depth = coor[kept2], depth[kept2]
    final_original_indices = original_indices[kept2]  # 최종 인덱스
    
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    
    valid_indices = torch.where(kept1)[0][kept2]
    
    return depth_map, coor, depth, valid_indices, final_original_indices  # 인덱스 추가 반환

def add_calibration(lidar2img, points_lidar): 
    points_img = points_lidar[:, :3].matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)

    return points_img

def add_calibration_adv(lidar2img, points_lidar): 
    lidar_points = points_lidar.tensor[:, :3]
    lidar_points_homo = torch.cat([lidar_points, torch.ones_like(lidar_points[:, :1])], dim=1)
    points_img = (lidar2img @ lidar_points_homo.T).T
    points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)

    return points_img

def add_calibration_adv2 (extrinsic, intrinsic, points_lidar) :
    lidar_points_xyz = points_lidar[:, :3]
    # lidar_points_intensity = points_lidar.tensor[:, 3:]
    lidar_points_homo = torch.cat([lidar_points_xyz, torch.ones_like(lidar_points_xyz[:, :1])], dim=1)
    KT = intrinsic @ extrinsic.T 
    points_img = (KT @ lidar_points_homo.T).T
    points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
    return points_img , KT

def add_mis_calibration_adv(lidar2img ,extrinsic, homo_intrinsic, points_lidar, max_r=1.0, max_t=0.1,generator=None,):
    device = extrinsic.device
    dtype = extrinsic.dtype
    intrinsic = homo_intrinsic[:3,:3] 
    # # 회전 각도 생성 (random)
    angles = torch.rand(3, device=device, dtype=dtype,generator=generator,) * 2 * max_r - max_r
    angles = angles * (torch.pi / 180.0)
    # # 회전 각도 생성 (고정 for test)
    # angles = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype) * (torch.pi / 180.0)  # 10도 → 라디안
    
    # 회전 행렬 생성 (Rx, Ry, Rz)
    Rx = torch.tensor([[1, 0, 0],
    [0, torch.cos(angles[0]), -torch.sin(angles[0])],
    [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device, dtype=dtype)

    Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
    [0, 1, 0],
    [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device, dtype=dtype)

    Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
    [torch.sin(angles[2]), torch.cos(angles[2]), 0],
    [0, 0, 1]], device=device, dtype=dtype)

    R_perturb = Rz @ Ry @ Rx
    # R_perturb = Rx @ Ry @ Rz

    # 2. 이동 벡터 생성 (delta_t)
    delta_t = (torch.rand(3, device=device, dtype=dtype,generator=generator,) * 2 * max_t - max_t)
    # 이동량 고정 (7.5cm)
    # delta_t = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)  # x,y,z 축 각각 7.5cm

    # # Extrinsic에 회전/이동 적용
    # RT_mis = extrinsic.clone()
    # # RT_mis[:3, :3] = R_perturb @ extrinsic[:3, :3]
    # RT_mis[:3, :3] = extrinsic[:3, :3] @ 
    
    # 3. Extrinsic Perturbation Matrix 생성 (검색 결과 [2]의 로직 적용)
    extrinsic_perturb = torch.eye(4, device=device, dtype=dtype)
    extrinsic_perturb[:3, :3] = R_perturb
    extrinsic_perturb[:3, 3] = delta_t

    # # 이동 벡터 덮어쓰기 (대입 방식으로 수정)
    # # RT_mis[:3, 3] = (torch.rand(3, device=device, dtype=dtype) * 2 * max_t - max_t)
    # delta_t = (torch.rand(3) * 2 * max_t - max_t)
    # RT_mis[:3, 3] = extrinsic[:3, 3] + delta_t

    # 4. 원본 Extrinsic에 Perturbation 적용 (RT_mis = extrinsic @ extrinsic_perturb)
    # RT_mis = extrinsic_perturb @ extrinsic  
    RT_mis = extrinsic @ extrinsic_perturb

    # --- intensity 분리 ---
    points_xyz = points_lidar[:, :3]           # [N,3]
    points_intensity = points_lidar[:, 3:]     # [N,1] (그 외 채널도 이 방식으로 분리)
    
    points_hom = torch.cat([points_xyz, torch.ones_like(points_xyz[:, :1])], dim=1)

    # 외란에 의해 변형된 라이다 포인트 클라우드
    perturbed_points_xyz = (extrinsic_perturb @ points_hom.T).T[:,:3]
    perturbed_points_lidar = torch.cat([perturbed_points_xyz, points_intensity], dim=1) 
    perturbed_points_hom = torch.cat([perturbed_points_xyz, torch.ones_like(perturbed_points_xyz[:, :1])], dim=1)

    # 디버깅용 출력 (확인용)
    if max_r == 0.0 and max_t == 0.0:
        assert torch.allclose(R_perturb, torch.eye(3, device=device)), "Rotation matrix is not identity!"
        assert torch.allclose(RT_mis[:3], extrinsic[:3]), "RT_mis does not match extrinsic!"
    # points_cam = (RT_mis @ points_hom.T).T  # [N,4]

    # # Step 3: 카메라 → 이미지 투영
    # projected = (intrinsic[:3,:3] @ points_cam[:, :3].T).T  # [N,3]

    # # 투영 행렬 계산
    # KT = intrinsic @ RT_mis
    # lidar2img_mis = intrinsic @ RT_mis.T
    lidar2img_mis = intrinsic @ RT_mis[:3, :]
    projected = (lidar2img_mis @ points_hom.T).T
    projected[:, :2] /= projected[:, 2:3] 
    
    # Perturbed 투영 (원래 extrinsic 사용)
    lidar2img_original = intrinsic @ extrinsic[:3, :]  # 기존 변환 행렬
    perturbed_projected = (lidar2img_original @ perturbed_points_hom.T).T
    perturbed_projected[:, :2] /= perturbed_projected[:, 2:3]

     # Perturbed 투영 (homo_intrinsic/extrinsic 사용)
    KT = homo_intrinsic @ extrinsic.T # 이게 lidar2img와 값이 같음
    points_img = (lidar2img @ perturbed_points_hom.T).T
    points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)

    points_img_mis_calibrated = projected[:, :3]
    perturbed_projected = perturbed_projected[:, :3]

    return points_img, perturbed_points_lidar, extrinsic_perturb, lidar2img_original ,lidar2img_mis


def add_mis_calibration_ori(extrinsic, intrinsic, points_lidar ,max_r=1.0, max_t=0.1):
    # 회전 각도 랜덤 생성 (deg 단위)
    max_angle = max_r  # 최대 회전 각도
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    
    # 회전 행렬 생성
    Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                [np.sin(rotz), np.cos(rotz), 0],
                [0, 0, 1]])
    Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                [0, 1, 0],
                [-np.sin(roty), 0, np.cos(roty)]])
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rotx), -np.sin(rotx)],
                [0, np.sin(rotx), np.cos(rotx)]])
    
    # 총 회전 행렬 생성
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # 이동 벡터 랜덤 생성
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    T = np.array([transl_x, transl_y, transl_z])

    # lidar2img 행렬에 mis-calibration 적용
    lidar2img_mis_calibrated = extrinsic.clone()
    lidar2img_mis_calibrated[:3, :3] = torch.tensor(R, dtype=torch.float32) @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += torch.tensor(T, dtype=torch.float32)

    # intrinsic 변환
    homo_intrinsic = torch.eye(4,dtype=torch.float32)
    # homo_intrinsic[:3,:3] = intrinsic
    homo_intrinsic = intrinsic
    KT = homo_intrinsic.matmul(lidar2img_mis_calibrated)

    # Mis-calibrated depth map 계산
    points_img_mis_calibrated = points_lidar.tensor[:, :3].matmul(KT[:3, :3].T) + KT[:3, 3].unsqueeze(0)
    points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
    # points_img_mis_calibrated = points_img_mis_calibrated.matmul(post_rots[cid].T) + post_trans[cid:cid + 1, :]
    # mis_calibrated_depth_map, uv , z = points2depthmap(points_img_mis_calibrated, imgs.shape[2], imgs.shape[3])
    
    return points_img_mis_calibrated , lidar2img_mis_calibrated, KT ,homo_intrinsic

def add_mis_calibration(extrinsic, intrinsic, points_lidar, max_r=1.0, max_t=0.1):
    """
    Apply mis-calibration to lidar points.
    
    Args:
    extrinsic (torch.Tensor): Shape (4, 4), extrinsic matrix
    intrinsic (torch.Tensor): Shape (3, 3) or (4, 4), intrinsic matrix
    points_lidar (torch.Tensor): Shape (N, 3) or (N, 4), lidar points
    max_r (float): Maximum rotation angle in degrees
    max_t (float): Maximum translation distance
    
    Returns:
    points_img_mis_calibrated (torch.Tensor): Mis-calibrated points in image space
    KT (torch.Tensor): Mis-calibrated projection matrix
    """
    device = extrinsic.device
    dtype = extrinsic.dtype

    # 1. 원본 Extrinsic의 4행 검증 및 강제 수정
    if not torch.allclose(extrinsic[3, :], torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)):
        extrinsic = extrinsic.clone()
        extrinsic[3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)
    
    # Generate random rotation angles
    angles = torch.rand(3, device=device, dtype=dtype) * 2 * max_r - max_r
    angles = angles * (torch.pi / 180.0)  # Convert to radians
    
    # Create rotation matrices
    Rx = torch.eye(3, device=device, dtype=dtype)
    Ry = torch.eye(3, device=device, dtype=dtype)
    Rz = torch.eye(3, device=device, dtype=dtype)
    
    if max_r > 0:
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                           [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device, dtype=dtype)
        
        Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                           [0, 1, 0],
                           [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device, dtype=dtype)
        
        Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                           [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                           [0, 0, 1]], device=device, dtype=dtype)
    
    R = Rz @ Ry @ Rx

    # Generate random translation
    T = torch.zeros(3, device=device, dtype=dtype)
    if max_t > 0:
        T = (torch.rand(3, device=device, dtype=dtype) * 2 * max_t - max_t)
    
    # Create homogeneous transformation matrix (4x4) for R and T
    RT = torch.eye(4, device=device, dtype=dtype)
    RT[:3, :3] = R
    RT[:3, 3] = T

    # Apply mis-calibration to extrinsic matrix
    lidar2img_mis_calibrated = extrinsic.clone()
    lidar2img_mis_calibrated[:3, :3] = R @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += T

    R_perturb = Rz @ Ry @ Rx
    T_perturb = (torch.rand(3, device=device, dtype=dtype) * 2 - 1) * max_t  # [-max_t, max_t]
    # 올바른 회전 적용: R_original @ R_perturb
    RT_mis = extrinsic.clone()
    RT_mis[:3, :3] = extrinsic[:3, :3] @ R_perturb  # 회전 적용
    RT_mis[:3, 3] += T_perturb  # 이동 적용
    RT_mis[3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)  # 4행 강제 설정
    
    # Ensure intrinsic is 4x4
    if intrinsic.shape == (3, 3):
        homo_intrinsic = torch.eye(4, device=device, dtype=dtype)
        homo_intrinsic[:3, :3] = intrinsic
    else:
        homo_intrinsic = intrinsic
    
    # KT = homo_intrinsic @ lidar2img_mis_calibrated
    KT = homo_intrinsic @ lidar2img_mis_calibrated.T
    ##### ref : K(T.t): [results['intrinsics'][i] @ results['extrinsics'][i].T 
    
    # Project points
    if points_lidar.shape[1] == 3:
        points_lidar = torch.cat([points_lidar, torch.ones(points_lidar.shape[0], 1, device=device, dtype=dtype)], dim=1)
    
    # points_img_mis_calibrated = (KT @ points_lidar.tensor.T).T
    # points_img_mis_calibrated = points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3]
    points_img_mis_calibrated = points_lidar.tensor[:, :3].matmul(KT[:3, :3].T) + KT[:3, 3].unsqueeze(0)
    points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
    
    return points_img_mis_calibrated, RT_mis, KT ,homo_intrinsic


def add_mis_calibration_cpu(extrinsic, intrinsic, points_lidar, max_r=1.0, max_t=0.1):
    """
    Apply mis-calibration to lidar points.
    
    Args:
    extrinsic (np.ndarray): Shape (4, 4), extrinsic matrix
    intrinsic (np.ndarray): Shape (3, 3) or (4, 4), intrinsic matrix
    points_lidar (np.ndarray): Shape (N, 3) or (N, 4), lidar points
    max_r (float): Maximum rotation angle in degrees
    max_t (float): Maximum translation distance
    
    Returns:
    points_img_mis_calibrated (np.ndarray): Mis-calibrated points in image space
    RT (np.ndarray): Mis-calibrated transformation matrix
    """
    # Generate random rotation angles
    angles = np.random.rand(3) * 2 * max_r - max_r
    angles = angles * (np.pi / 180.0)  # Convert to radians
    
    # Create rotation matrices
    Rx = np.eye(3)
    Ry = np.eye(3)
    Rz = np.eye(3)
    
    if max_r > 0:
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Generate random translation
    T = np.zeros(3)
    if max_t > 0:
        T = (np.random.rand(3) * 2 * max_t - max_t)
    
    # Create homogeneous transformation matrix (4x4) for R and T
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = T

    # Apply mis-calibration to extrinsic matrix
    lidar2img_mis_calibrated = extrinsic.copy()
    lidar2img_mis_calibrated[:3, :3] = R @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += T
    
    # Ensure intrinsic is 4x4
    if intrinsic.shape == (3, 3):
        homo_intrinsic = np.eye(4)
        homo_intrinsic[:3, :3] = intrinsic
    else:
        homo_intrinsic = intrinsic
    
    KT = homo_intrinsic @ lidar2img_mis_calibrated.T
    
    # Project points
    if points_lidar.shape[1] == 3:
        points_lidar = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
    
    points_img_mis_calibrated = np.dot(points_lidar[:, :3], KT[:3, :3].T) + KT[:3, 3]
    points_img_mis_calibrated = np.hstack([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]])
    
    return points_img_mis_calibrated, RT


def apply_random_rt_to_point_cloud(point_cloud, max_r=7.5, max_t=0.20):
    # 회전 각도 랜덤 생성 (deg 단위)
    max_angle = max_r  # 최대 회전 각도
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)

    # 회전 행렬 생성
    Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                   [np.sin(rotz), np.cos(rotz), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                   [0, 1, 0],
                   [-np.sin(roty), 0, np.cos(roty)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotx), -np.sin(rotx)],
                   [0, np.sin(rotx), np.cos(rotx)]])

    # 총 회전 행렬 생성
    R = np.dot(Rz, np.dot(Ry, Rx))

    # 이동 벡터 랜덤 생성
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    T = np.array([transl_x, transl_y, transl_z])
    
    # 포인트 클라우드에 회전 및 이동 변환 적용
    rotated_points = np.dot(point_cloud, R.T)
    transformed_points = rotated_points + T

    return transformed_points

def project_points_to_image(points, camera_intrinsics, camera_extrinsics, image_shape):
    """
    포인트 클라우드를 카메라 이미지에 투영하는 함수
    points: (N, 3) 크기의 포인트 클라우드 텐서
    camera_intrinsics: (3, 3) 크기의 카메라 내부 행렬
    camera_extrinsics: (4, 4) 크기의 카메라 외부 행렬
    image_shape: (height, width) 이미지 크기
    """
    # # 포인트 클라우드에 동차 좌표 추가
    # ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
    # points_homogeneous = torch.cat([points, ones], dim=1)  # (N, 4)

    # # 카메라 좌표계로 변환
    # points_camera = (camera_extrinsics @ points_homogeneous.T).T[:, :3]

    # 이미지 평면으로 투영
    points_image = (camera_intrinsics @ points.T).T

    # 동차 좌표에서 유클리드 좌표로 변환
    uv = points_image[:, :2] / points_image[:, 2:3]
    z = points_image[:, 2]  # z 값 추출

    # 이미지 범위 내의 포인트만 필터링
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & \
           (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0])
    
    points_image = points_image[mask]
    uv = uv[mask]
    z  = z[mask]
    
    return points_image ,uv ,z

def dense_map_gpu_optimized(Pts, n, m, grid):
    device = Pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8  # 작은 값 추가하여 0으로 나누는 상황 방지
    # import pdb; pdb.set_trace()
    # 초기 텐서를 GPU로 이동
    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mD = torch.zeros((m, n), dtype=torch.float32, device=device)

    mX_idx = Pts[1].clone().detach().to(dtype=torch.int32, device=device)
    mY_idx = Pts[0].clone().detach().to(dtype=torch.int32, device=device)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)

    # KmX = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmY = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmD = torch.zeros((ng, ng), dtype=torch.float32, device=device)

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + j
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]

    S = torch.zeros_like(KmD[0, 0], device=device)
    Y = torch.zeros_like(KmD[0, 0], device=device)

    for i in range(ng):
        for j in range(ng):
            # s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2)
            s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2 + epsilon)
            Y += s * KmD[i, j]
            S += s

    S[S == 0] = 1
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    # return out.cpu()  # 최종 결과를 CPU로 이동
    return out # 최종 결과를 GPU

def distance_adaptive_depth_completion(pts, n, m, grid):
    # Similar to original function but with distance-adaptive parameters
    
    # Extract depth information
    depths = pts[2]
    
    # Create multiple grid sizes based on distance
    grid_sizes = torch.ones_like(depths, device=pts.device)
    # Closer points use smaller grids for higher detail
    grid_sizes[depths < 10] = grid // 4
    # Further points use larger grids for better filling
    grid_sizes[depths > 30] = grid
    grid_sizes[depths > 50] = grid // 4
    
    # Process each distance zone separately
    result = torch.zeros((m, n), dtype=torch.float32, device=pts.device)
    
    for g in torch.unique(grid_sizes):
        mask = (grid_sizes == g)
        pts_subset = pts[:, mask]
        temp_result = dense_map_gpu_optimized(pts_subset, n, m, int(g))
        # Combine results, prioritizing smaller grid results (higher detail)
        valid_mask = (temp_result > 0)
        result[valid_mask] = temp_result[valid_mask]
    
    return result

# 포인트 클라우드를 전처리하여 z값 반전
def preprocess_points(pts):
    # z값의 최대, 최소 확인
    z_min = pts[2].min()
    z_max = pts[2].max()
    
    # z값 반전 (최대값+최소값-z)
    pts_normalized = pts.clone()
    pts_normalized[2] = z_max + z_min - pts[2]
    
    return pts_normalized

from skimage import feature
def canny_edge_detection(rgb_img, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
    """
    Perform Canny edge detection on an RGB image.

    Parameters:
        rgb_img (torch.Tensor): Input RGB image as a PyTorch tensor of shape (3, H, W).
        sigma (float): Standard deviation of the Gaussian filter used in the Canny algorithm.
        low_threshold (float): Lower bound for hysteresis thresholding.
        high_threshold (float): Upper bound for hysteresis thresholding.

    Returns:
        torch.Tensor: Binary edge map as a PyTorch tensor of shape (H, W).
    """
    # Convert PyTorch tensor to NumPy array and grayscale
    rgb_img_np = rgb_img.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, 3)
    grayscale_img = np.dot(rgb_img_np[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale conversion

    # Apply Canny edge detection
    edges = feature.canny(
        image=grayscale_img,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )

    # Convert edge map back to PyTorch tensor
    edges_tensor = torch.tensor(edges, dtype=torch.float32, device=rgb_img.device)

    return edges_tensor


def edge_aware_bilateral_filter(pts, rgb_img, n, m, grid):
    """
    Perform edge-aware bilateral filtering on a sparse depth map using RGB image edges.

    Parameters:
        pts (torch.Tensor): Input points as a tensor of shape (3, N), where N is the number of points.
                            pts[0] = y-coordinates, pts[1] = x-coordinates, pts[2] = depth values.
        rgb_img (torch.Tensor): Input RGB image as a PyTorch tensor of shape (3, H, W).
        n (int): Width of the output depth map.
        m (int): Height of the output depth map.
        grid (int): Grid size for the bilateral filter.

    Returns:
        torch.Tensor: Densified depth map as a PyTorch tensor of shape (m, n).
    """
    device = pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8  # Small value to avoid division by zero

    # Initialize tensors on GPU
    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mD = torch.zeros((m, n), dtype=torch.float32, device=device)

    mX_idx = pts[1].clone().detach().to(dtype=torch.int64)
    mY_idx = pts[0].clone().detach().to(dtype=torch.int64)

    mX[mX_idx, mY_idx] = pts[0] - torch.round(pts[0])
    mY[mX_idx, mY_idx] = pts[1] - torch.round(pts[1])
    mD[mX_idx, mY_idx] = pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)

    for i in range(ng):
        for j in range(ng):
            KmX[i][j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i][j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + j
            KmD[i][j] = mD[i: (m - ng + i), j: (n - ng + j)]

    # Detect edges in the RGB image
    edges = canny_edge_detection(rgb_img)

    # Resize edges to match spatial dimensions of KmD
    edges_resized = torch.nn.functional.interpolate(
        edges.unsqueeze(0).unsqueeze(0), size=(m - ng, n - ng), mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)

    S = torch.zeros_like(KmD[0][0], device=device)
    Y = torch.zeros_like(KmD[0][0], device=device)

    sigma_range = 0.1  # Range weight parameter
    edge_weight = 10.0  # Weight boost for edges

    for i in range(ng):
        for j in range(ng):
            # Calculate spatial weight based on distance
            s_spatial = 1 / torch.sqrt(KmX[i][j] ** 2 + KmY[i][j] ** 2 + epsilon)

            # Calculate range weight based on depth difference
            center_depth = KmD[ng // 2][ng // 2]
            s_range = torch.exp(-torch.abs(KmD[i][j] - center_depth) / sigma_range)

            # Incorporate edge information into weights
            s_edge = torch.ones_like(s_spatial)
            s_edge[edges_resized > 0] *= edge_weight

            # Combine weights
            s = s_spatial * s_range * s_edge
            Y += s * KmD[i][j]
            S += s

    S[S == 0] = 1  # Avoid division by zero
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S

    return out

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()        # tensor -> numpy
    # disp_np = disp
    # vmax = np.percentile(disp_np, 95)
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    # colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32)
    colormapped_tensor = torch.from_numpy(colormapped_im).to(dtype=torch.float32)
    return colormapped_tensor

def find_exact_correspondences(valid_indices_gt, valid_indices, gt_uv, uv):
    """정확한 대응쌍을 찾는 함수"""
    device = gt_uv.device
    
    # GPU 텐서를 CPU로 변환
    valid_gt_cpu = valid_indices_gt.cpu().numpy()
    valid_mis_cpu = valid_indices.cpu().numpy()
    
    matched_pairs = []
    
    # 각 GT 인덱스에 대해 대응되는 MIS 인덱스 찾기
    for i, gt_idx in enumerate(valid_gt_cpu):
        # GT 인덱스가 MIS에도 존재하는지 확인
        mis_positions = np.where(valid_mis_cpu == gt_idx)[0]
        
        if len(mis_positions) > 0:
            # 대응되는 첫 번째 위치 사용
            mis_pos = mis_positions[0]
            
            # 대응쌍 저장: (gt_position, mis_position, common_index)
            matched_pairs.append((i, mis_pos, gt_idx))
    
    if len(matched_pairs) == 0:
        return torch.empty((0, 4), device=device)
    
    # 매칭된 좌표 추출
    matched_gt_indices = [pair[0] for pair in matched_pairs]
    matched_mis_indices = [pair[1] for pair in matched_pairs]
    
    matched_gt_uv = gt_uv[matched_gt_indices]
    matched_mis_uv = uv[matched_mis_indices]
    
    # 4열로 연결: [u_gt, v_gt, u_mis, v_mis]
    matched_uv_set = torch.cat([matched_gt_uv, matched_mis_uv], dim=1)
    
    return matched_uv_set

# def trim_corrs(points ,num_kp=30000):
#     length = points.shape[0]
# #         print ("number of keypoint before trim : {}".format(length))
#     if length >= num_kp:
#         # mask = np.random.choice(length, num_kp)
#         mask = torch.randperm(length)[:num_kp]
#         return points[mask]
#     else:
#         # mask = np.random.choice(length, num_kp - length)
#         mask = torch.randint(0, length, (num_kp - length,))
#         # return np.concatenate([points, points[mask]], axis=0)
#         return torch.cat([points, points[mask]], dim=0)

def trim_corrs(points, num_kp=30000):
    length = points.shape[0]
    # print("number of keypoint before trim : {}".format(length))
    if length == 0:
        # points의 feature dimension을 유지하며 랜덤 keypoint 생성
        # 예: points shape == (N, D)일 때 D 추출
        feature_dim = points.shape[1] if points.ndim > 1 else 1
        # points의 dtype, device도 맞춰줌
        random_points = torch.rand((num_kp, feature_dim), dtype=points.dtype, device=points.device)
        return random_points
    elif length >= num_kp:
        mask = torch.randperm(length)[:num_kp]
        return points[mask]
    else:
        mask = torch.randint(0, length, (num_kp - length,))
        return torch.cat([points, points[mask]], dim=0)


def enhanced_geometric_propagation(depth_map, iterations=10, base_threshold=1.0):
    """
    다방향 전파와 적응형 임계값을 사용한 고밀도 깊이 맵 생성 알고리즘
    """
    # 입력 차원 처리
    orig_dim = len(depth_map.shape)
    if orig_dim == 2:
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
    elif orig_dim == 3:
        depth_map = depth_map.unsqueeze(1)  # [B,H,W] → [B,1,H,W]
    
    # 결과 복사본 생성 및 원본 값 보존
    result = depth_map.clone()
    original_valid = (depth_map > 0).float()
    
    # 8방향 전파 정의 (4개 직교 + 4개 대각선)
    directions = [
        (0, 1),    # 오른쪽
        (0, -1),   # 왼쪽
        (1, 0),    # 아래
        (-1, 0),   # 위
        (1, 1),    # 오른쪽-아래
        (-1, -1),  # 왼쪽-위
        (1, -1),   # 왼쪽-아래
        (-1, 1)    # 오른쪽-위
    ]
    
    for iter_idx in range(iterations):
        # 반복이 진행됨에 따라 임계값 점진적 감소 (첫 반복 = 관대, 마지막 반복 = 엄격)
        current_threshold = base_threshold * (1.0 - 0.5 * iter_idx / iterations)
        
        # 각 방향 처리
        for dy, dx in directions:
            # 대각선 방향은 약간 높은 임계값 사용
            dir_threshold = current_threshold * (1.4 if abs(dy) + abs(dx) > 1 else 1.0)
            
            # 소스와 대상 픽셀의 슬라이스 인덱스 계산
            if dx > 0:
                x_src, x_tgt = slice(0, -1), slice(1, None)
            elif dx < 0:
                x_src, x_tgt = slice(1, None), slice(0, -1)
            else:
                x_src = x_tgt = slice(None)
                
            if dy > 0:
                y_src, y_tgt = slice(0, -1), slice(1, None)
            elif dy < 0:
                y_src, y_tgt = slice(1, None), slice(0, -1)
            else:
                y_src = y_tgt = slice(None)
            
            # 이동이 없는 방향은 건너뛰기
            if y_src == slice(None) and x_src == slice(None):
                continue
            
            # 소스 및 대상 값 추출
            src_depth = result[..., y_src, x_src]
            tgt_depth = result[..., y_tgt, x_tgt]
            
            # 유효한 픽셀 결정
            src_valid = (src_depth > 0).float()
            tgt_valid = (tgt_depth > 0).float()
            
            # 깊이 차이 계산
            depth_diff = torch.abs(src_depth - tgt_depth)
            
            # 전파 마스크 생성:
            # 1. 유효한 이웃에서 빈 픽셀 채우기
            # 2. 차이가 임계값 미만인 경우 업데이트
            update_mask = ((tgt_valid == 0) & (src_valid > 0)) | \
                          ((tgt_valid > 0) & (src_valid > 0) & (depth_diff < dir_threshold))
            
            # 전파 적용
            result[..., y_tgt, x_tgt] = torch.where(
                update_mask, 
                src_depth,
                result[..., y_tgt, x_tgt]
            )
    
    # 원본 유효 측정값 보존
    result = torch.where(original_valid > 0, depth_map, result)
    
    # 원래 차원으로 반환
    if orig_dim == 2:
        return result.squeeze()
    elif orig_dim == 3:
        return result.squeeze(1)
    else:
        return result

def direction_aware_completion(sparse_depth, num_directions=8, max_radius=50):
    result = sparse_depth.clone()
    h, w = sparse_depth.shape
    device = sparse_depth.device
    
    # 방향 벡터 정의
    angles = torch.linspace(0, 2*np.pi, num_directions+1)[:-1]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(device)
    
    # 각 픽셀에 대해
    for y in range(h):
        for x in range(w):
            if result[y, x] > 0:  # 이미 값이 있으면 건너뜀
                continue
                
            valid_values = []
            weights = []
            
            # 각 방향으로 탐색
            for dir_vec in directions:
                dx, dy = dir_vec
                for r in range(1, max_radius+1):
                    nx, ny = int(x + dx*r), int(y + dy*r)
                    
                    # 경계 체크
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        break
                        
                    # 유효한 값 찾음
                    if result[ny, nx] > 0:
                        valid_values.append(result[ny, nx])
                        # 거리 기반 가중치 (거리 제곱의 역수)
                        weights.append(1/(r*r))
                        break
            
            # 가중 평균 계산
            if valid_values:
                weights = torch.tensor(weights, device=device)
                valid_values = torch.tensor(valid_values, device=device)
                result[y, x] = torch.sum(weights * valid_values) / torch.sum(weights)
    
    return result

def direction_aware_bilateral_filter(Pts, n, m, grid):
    device = Pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8
    
    # Initialize sparse depth map
    sparse_depth = torch.zeros((m, n), dtype=torch.float32, device=device)
    x_indices = Pts[1].to(dtype=torch.int64)
    y_indices = Pts[0].to(dtype=torch.int64)
    sparse_depth[x_indices, y_indices] = Pts[2]
    
    # Create kernel tensors
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    
    for i in range(ng):
        for j in range(ng):
            KmD[i, j] = sparse_depth[i: (m - ng + i), j: (n - ng + j)]
    
    # Output accumulators
    Y = torch.zeros_like(KmD[0, 0], device=device)
    S = torch.zeros_like(KmD[0, 0], device=device)
    
    # Parameters for anisotropic filtering
    sigma_horizontal = grid / 1.0  # Larger sigma for horizontal direction
    sigma_vertical = grid / 3.0    # Smaller sigma for vertical direction
    
    for i in range(ng):
        for j in range(ng):
            # Anisotropic spatial weight - emphasizes horizontal connections
            y_diff = (i - grid)**2 / (2 * sigma_vertical**2)
            x_diff = (j - grid)**2 / (2 * sigma_horizontal**2)
            
            # Convert scalar to tensor before using torch.exp
            diff_tensor = torch.tensor(y_diff + x_diff, device=device)
            spatial_weight = torch.exp(-diff_tensor)
            
            # Only consider pixels with positive depth
            valid_mask = KmD[i, j] > 0
            Y[valid_mask] += spatial_weight * KmD[i, j][valid_mask]
            S[valid_mask] += spatial_weight
    
    # Normalize
    output_depth = torch.zeros((m, n), dtype=torch.float32, device=device)
    valid_S = S > epsilon
    output_depth[grid + 1: -grid, grid + 1: -grid][valid_S] = Y[valid_S] / S[valid_S]
    
    return output_depth

def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    3x3 회전 행렬을 axis-angle 벡터로 변환합니다.
    """
    # 클램프를 사용하여 acos의 입력 범위를 [-1, 1]로 제한
    trace = torch.clamp(R.trace(), -1.0, 3.0)
    angle = torch.acos((trace - 1.0) / 2.0)
    
    # 각도가 0에 가까우면 (단위 행렬) 축은 의미 없으므로 영벡터 반환
    if angle < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    
    # 축 계산
    axis_x = R[2, 1] - R[1, 2]
    axis_y = R[0, 2] - R[2, 0]
    axis_z = R[1, 0] - R[0, 1]
    
    axis = torch.tensor([axis_x, axis_y, axis_z], device=R.device, dtype=R.dtype)
    axis = axis / (2.0 * torch.sin(angle))
    
    return angle * axis