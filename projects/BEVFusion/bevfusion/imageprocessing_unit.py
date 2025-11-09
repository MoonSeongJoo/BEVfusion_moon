import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import matplotlib as mpl
import matplotlib.cm as cm
import scipy
import skimage
# from pypardiso import spsolve
from PIL import Image , ImageDraw
# from COTR.utils import utils
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
# from geomloss import SamplesLoss
import os
import cv2
import matplotlib.patches as patches

def visualize_bboxes(img_tensor, proposal_list, output_path='output.png', dpi=150):
    """
    6개 카메라 뷰에 2D 바운딩 박스 시각화 후 PNG 저장
    
    Args:
        img_tensor (Tensor): [6, 3, H, W] 형태의 이미지 텐서
        proposal_list (list): 카메라별 바운딩 박스 리스트
        output_path (str): 출력 파일 경로
        dpi (int): 이미지 해상도 (기본 150)
    """
    num_cams = img_tensor.shape[0]
    fig, axs = plt.subplots(2, 3, figsize=(24, 12), dpi=dpi)
    
    for cam_idx in range(num_cams):
        row = cam_idx // 3
        col = cam_idx % 3
        
        # 이미지 텐서 처리
        img = img_tensor[cam_idx].permute(1, 2, 0).cpu().numpy()
        if img.max() <= 1.0:  # 정규화 여부 확인
            img = (img * 255).astype(np.uint8)
        
        # 서브플롯 설정
        ax = axs[row, col]
        ax.imshow(img)
        ax.set_title(f'Camera {cam_idx}', fontsize=8)
        ax.axis('off')
        
        # 바운딩 박스 그리기
        if cam_idx < len(proposal_list):
            for bbox in proposal_list[cam_idx]:
                x1, y1, x2, y2 ,_,_ = bbox.cpu()
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle(
                    (x1, y1), width, height,
                    linewidth=1.5, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 메모리 해제


# def display_depth_maps(imgs, mis_calibrated_depth_map, sbs_img):
#     # """
#     # 이미지 위에 depth map과 mis_calibrated depth map을 오버레이하여 디스플레이하는 함수
#     # """

#     for cid in range(imgs.shape[0]):

#         img_np = imgs.squeeze()[cid].cpu().detach().numpy()
#         # depth_gt_np = depth_map.squeeze()[cid].cpu().detach().numpy() 
#         depth_mis_np = mis_calibrated_depth_map.squeeze()[cid].cpu().detach().numpy()
#         sbs_img_np = sbs_img.squeeze()[cid].cpu().detach().numpy()
#         img_np = np.transpose(img_np,(1,2,0))
#         # depth_gt_np = np.transpose(depth_gt_np,(1,2,0))
#         depth_mis_np = np.transpose(depth_mis_np,(1,2,0))
#         sbs_img_np = np.transpose(sbs_img_np,(1,2,0))
#         # # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
#         # if img_np.dtype == np.float32 or img_np.dtype == np.float64:
#         #     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
#         #     # depth_gt_np = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min())
#         #     depth_mis_np = (depth_mis_np - depth_mis_np.min()) / (depth_mis_np.max() - depth_mis_np.min())
#         #     sbs_img_np = (sbs_img_np - sbs_img_np.min()) / (sbs_img_np.max() - sbs_img_np.min())

#         # input display
#         ###### display input signal #########        
#         plt.figure(figsize=(20, 20))
#         plt.subplot(311)
#         plt.imshow(img_np)
#         plt.title("camera_input", fontsize=15)
#         plt.axis('off')

#         # plt.subplot(312)
#         # plt.imshow( depth_gt_np, cmap='magma')
#         # plt.title("calibrated_lidar_input", fontsize=15)
#         # plt.axis('off') 

#         plt.subplot(312)
#         plt.imshow( depth_mis_np, cmap='magma')
#         plt.title("mis-calibrated_lidar_input", fontsize=15)
#         plt.axis('off')

#         plt.subplot(313)
#         plt.imshow( sbs_img_np)
#         plt.title("sbs_img_input", fontsize=15)
#         plt.axis('off')
        
#         plt.tight_layout(pad=0)  # 여백 제거
#         plt.savefig(f'raw_input_image_{cid+1}.png', dpi=300, bbox_inches='tight')
#         plt.close('all')
    
#     print ('display end')
#     # ############ end of display input signal ###################

def display_depth_maps(imgs, mis_calibrated_depth_map, sbs_img):
    # 5D 텐서에서 배치 크기와 카메라 수를 가져옵니다.
    batch_size, num_cam, _, _, _ = imgs.shape
    
    # 1. 배치 크기만큼 반복
    for b in range(batch_size):
        # 2. 카메라 수만큼 반복
        for cid in range(num_cam):
            # .squeeze(0) 대신 [b, cid]로 정확히 인덱싱합니다.
            img_np = imgs[b, cid].cpu().detach().numpy()
            depth_mis_np = mis_calibrated_depth_map[b, cid].cpu().detach().numpy()
            sbs_img_np = sbs_img[b, cid].cpu().detach().numpy()
            
            # Matplotlib으로 시각화하기 위해 차원을 (C, H, W) -> (H, W, C)로 변경
            img_np = np.transpose(img_np, (1, 2, 0))
            depth_mis_np = np.transpose(depth_mis_np, (1, 2, 0))
            sbs_img_np = np.transpose(sbs_img_np, (1, 2, 0))

            # (중요) 카메라 이미지가 모델 입력용으로 정규화된 상태이므로,
            # 시각화를 위해 픽셀 값을 0.0 ~ 1.0 범위로 다시 스케일링합니다.
            img_min, img_max = np.min(img_np), np.max(img_np)
            if img_max > img_min:
                img_np = (img_np - img_min) / (img_max - img_min)

            # Figure 및 subplot 생성
            plt.figure(figsize=(20, 20))
            
            # Camera image
            plt.subplot(311)
            plt.imshow(img_np)
            plt.title(f"Batch {b}, Cam {cid+1}: Camera Image", fontsize=15)
            plt.axis('off')

            # Mis-calibrated depth map
            plt.subplot(312)
            # 컬러맵이 있는 깊이맵은 채널이 1개일 수 있으므로 squeeze()로 차원 정리
            plt.imshow(depth_mis_np.squeeze(), cmap='magma')
            plt.title(f"Batch {b}, Cam {cid+1}: Mis-calibrated LiDAR Depth Map", fontsize=15)
            plt.axis('off')

            # Side-by-side comparison image
            plt.subplot(313)
            plt.imshow(sbs_img_np)
            plt.title(f"Batch {b}, Cam {cid+1}: Sensor Fusion Comparison", fontsize=15)
            plt.axis('off')

            # 파일 저장 시 덮어쓰지 않도록 배치와 카메라 인덱스를 모두 포함
            plt.tight_layout(pad=0)
            plt.savefig(f'calibration_visualization_batch{b}_cam{cid+1}.png', dpi=300, bbox_inches='tight')
            plt.close('all')
    
    print(f'Visualization completed for {batch_size} batches.')

def normalize_uvz_points(points_lidar2img):
    
    normal_points_lidar2img = points_lidar2img.clone()
    normal_points_lidar2img[:, 0] = normal_points_lidar2img[:, 0]/1280
    normal_points_lidar2img[:, 1] = normal_points_lidar2img[:, 1]/192
    if normal_points_lidar2img[:, 2].numel() > 0:
        normal_points_lidar2img[:, 2] = (normal_points_lidar2img[:, 2]-torch.min(normal_points_lidar2img[:, 2]))\
            /(torch.max(normal_points_lidar2img[:, 2]) - torch.min(normal_points_lidar2img[:, 2]))
    else :
        normal_points_lidar2img[:, 2] = (normal_points_lidar2img[:, 2]-0)/(80 - 0)

    return normal_points_lidar2img

def normalize_uv_points(points_lidar2img):
    
    normal_points_lidar2img = points_lidar2img.clone()
    normal_points_lidar2img[..., 0] = normal_points_lidar2img[..., 0]/1280
    normal_points_lidar2img[..., 1] = normal_points_lidar2img[..., 1]/192
   
    return normal_points_lidar2img

def scale_uvz_points(uvz_tensor, original_size=(900, 1600), target_size=(192, 640)):
    """
    UVZ 좌표를 이미지 스케일링 비율에 맞게 변환
    Args:
        uvz_tensor: (N, 3) 형태의 텐서 [u, v, z]
        original_size: (H, W) 원본 이미지 크기
        target_size: (H, W) 타겟 이미지 크기
    Returns:
        스케일링된 (N, 3) UVZ 텐서
    """
    # 스케일링 계수 계산 (높이, 너비)
    scale_h = target_size[0] / original_size[0]  # 192/512 = 0.375
    scale_w = target_size[1] / original_size[1]  # 640/1408 ≈ 0.4545
    
    # UV 좌표 스케일링 (깊이 z는 변경 없음)
    scaled_uvz = uvz_tensor.clone()
    scaled_uvz[..., 0] *= scale_w  # u 좌표(너비 방향) 스케일
    scaled_uvz[..., 1] *= scale_h  # v 좌표(높이 방향) 스케일
    
    return scaled_uvz

def inverse_scale_uvz_points(scaled_uvz, original_size=(928, 1600), target_size=(192, 640)):
    """
    스케일링된 UVZ 좌표를 원본 크기로 역변환
    Args:
        scaled_uvz: (N, number of query , 3) 형태의 텐서 [scaled_u, scaled_v, z]
        original_size: (H, W) 원본 이미지 크기 (스케일링 전)
        target_size: (H, W) 타겟 이미지 크기 (스케일링 후)
    Returns:
        역변환된 (N,number of query, 3) UVZ 텐서
    """
    # 역스케일링 계수 계산 (높이, 너비)
    inv_scale_h = original_size[0] / target_size[0]  # 512/192 ≈ 2.6667
    inv_scale_w = original_size[1] / target_size[1]  # 1408/640 = 2.2
    
    # 좌표 복원
    original_uvz = scaled_uvz.clone()
    original_uvz[... , 0] *= inv_scale_w  # u 좌표 복원
    original_uvz[... , 1] *= inv_scale_h  # v 좌표 복원
    
    return original_uvz

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    # pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    # cam_intrinsic = cam_calib.numpy()
    # pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    pc_rotated = pc_rotated[:3, :].detach()
    cam_intrinsic = torch.tensor(cam_calib, dtype=torch.float32).cuda()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.t(), cam_intrinsic)
    
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0] ) & (pcl_z > 0)
    # mask1 = (pcl_uv[:, 1] < 188)

    pcl_uv_no_mask = pcl_uv
    # pcl_z_no_mask = pcl_z
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int64)
    # pcl_uv = pcl_uv.astype(np.uint32)
    # pcl_uv_no_mask  = pcl_uv_no_mask.astype(np.uint32) 
    
    pcl_z = pcl_z.reshape(-1, 1)

    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device=pcl_uv_no_mask.device)[mask]

    # points_index = np.arange(pcl_uv_no_mask.shape[0])[mask]
    # points_index1 = np.arange(pcl_uv_no_mask.shape[0])[mask1]

    return depth_img, pcl_uv , pcl_z , points_index 

def lidar_project_depth_nuscenes(nusc,lidar_points, cam_data, real_shape):
    
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc_points = lidar_points.cpu().numpy()
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc_points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc_points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # points = view_points(lidar_points.points, np.array(cs_record['camera_intrinsic']), normalize=True)

    # 투영된 점들이 이미지 안에 있는지 확인하고, 그렇다면 그 위치에 따라 점들을 그립니다.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < real_shape[1]) # 이미지의 width 안에 있는지 확인
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < real_shape[0]) # 이미지의 height 안에 있는지 확인

    pcl_uv = points.transpose(1,0)[:,:2]
    pcl_z = points.transpose(1,0)[:,2]
    pcl_uv_no_mask = pcl_uv
    pcl_uv = pcl_uv[mask]    
    # pcl_z = pcl_z[mask]
    pcl_z = depths[mask]
    pcl_uv = torch.tensor(pcl_uv, dtype=torch.int32).cuda()
    pcl_z = torch.tensor(pcl_z, dtype=torch.float32).cuda()
    pcl_z = pcl_z.reshape(-1, 1)
    
    depth_img = np.zeros((real_shape[0], real_shape[1], 1))
    depth_img = torch.from_numpy(depth_img.astype(np.float32)).cuda()
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z  
    depth_img = depth_img.permute(2, 0, 1)
    points_index = torch.arange(pcl_uv_no_mask.shape[0], device='cuda')[mask]

    return depth_img, pcl_uv , pcl_z , points_index

def transform_gt(nusc, lidar_data, lidar_points ,cam_data) :
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    lidar_cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_points.rotate(Quaternion(lidar_cs_record['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_cs_record['translation']))

    # Second step: transform from ego to the global frame.
    lidar_poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_points.rotate(Quaternion(lidar_poserecord['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
    lidar_points.translate(-np.array(poserecord['translation']))
    lidar_points.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_points.translate(-np.array(cs_record['translation']))
    lidar_points.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    return lidar_points
    
# def trim_corrs(in_corrs, num_kp=100):
#     length = in_corrs.shape[0]
# #         print ("number of keypoint before trim : {}".format(length))
#     if length >= num_kp:
#         mask = np.random.choice(length, num_kp)
#         return in_corrs[mask]
#     elif length==0:
#         return np.random.rand(num_kp, 3)
#     else:
#         mask = np.random.choice(length, num_kp - length)
#         return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

def trim_corrs(in_corrs, num_kp=100):
    device = in_corrs.device  # 원본 텐서 디바이스 유지
    length = in_corrs.shape[0]
    
    if length >= num_kp:
        # 무작위 선택 (비복원 추출)
        mask = torch.randperm(length, device=device)[:num_kp]
        return in_corrs[mask]
    elif length == 0:
        # 0~1 범위 랜덤 텐서 생성 [num_kp, 6]
        return torch.rand(num_kp, 3, device=device)
    else:
        # 부족분 채우기 (복원 추출)
        mask = torch.randint(0, length, (num_kp - length,), device=device)
        return torch.cat([in_corrs, in_corrs[mask]], dim=0)


def batched_trim_corrs(in_corrs, num_kp=200):
    """
    배치 처리된 correspondence 포인트 트리밍 함수
    Args:
        in_corrs: [batch_size, num_points, 4] 형태의 입력 텐서
        num_kp: 최종 출력 포인트 수 (기본값 100)
    Returns:
        [batch_size, num_kp, 4] 형태의 출력 텐서
    """
    device = in_corrs.device
    batch_size = in_corrs.size(0)
    results = []

    for i in range(batch_size):
        batch_data = in_corrs[i]  # [num_points, 4]
        num_points = batch_data.size(0)

        # Case 1: 충분한 포인트 (> num_kp)
        if num_points >= num_kp:
            idx = torch.randperm(num_points, device=device)[:num_kp]
            trimmed = batch_data[idx]
        
        # Case 2: 빈 입력
        elif num_points == 0:
            trimmed = torch.rand(num_kp, 4, device=device)
        
        # Case 3: 부족한 포인트 (< num_kp)
        else:
            # 기존 포인트 유지
            trimmed = batch_data
            
            # 추가 샘플링 (복원 추출)
            extra_needed = num_kp - num_points
            idx = torch.randint(0, num_points, (extra_needed,), device=device)
            trimmed = torch.cat([trimmed, batch_data[idx]], dim=0)
        
        results.append(trimmed)

    return torch.stack(results, dim=0)


# def process_queries(corrs, sbs_img, num_points=100):
#     device = corrs.device
    
#     # 1. 고유 카메라 인덱스 추출 및 정렬
#     unique_cams, counts = torch.unique(corrs[:, 0], 
#                                      sorted=False, 
#                                      return_counts=True)
    
#     # 2. 이미지 선택
#     selected_imgs = sbs_img[unique_cams.long()]
    
#     # 3. 카메라별 쿼리 처리
#     grouped_queries = []
#     camera_indices = []
    
#     for cam_idx, cam in enumerate(unique_cams):
#         # 현재 카메라 쿼리 추출
#         mask = (corrs[:, 0] == cam)
#         cam_queries = corrs[mask, 1:]  # [N, 4] (obj_id, u, v, z)
        
#         # 트리밍/패딩
#         if cam_queries.size(0) >= num_points:
#             idx = torch.randperm(cam_queries.size(0))[:num_points]
#             selected = cam_queries[idx]
#         else:
#             idx = torch.randint(0, cam_queries.size(0), (num_points - cam_queries.size(0),))
#             selected = torch.cat([cam_queries, cam_queries[idx]], dim=0)
        
#         # 카메라 인덱스 태깅 및 저장
#         tagged_queries = torch.cat([
#             torch.full((num_points, 1), cam_idx, device=device, dtype=torch.float),
#             selected
#         ], dim=1)  # [num_points, 5] (local_cam_idx,ob_id, u, v, z)
        
#         grouped_queries.append(tagged_queries)
#         camera_indices.append(cam)  # 원본 카메라 인덱스 저장

#     # 4. 배치 차원 생성 및 원본 카메라 인덱스 반환
#     processed_queries = torch.stack(grouped_queries, dim=0)  # [B_sel, 100, 4]
#     original_camera_ids = torch.stack(camera_indices)        # [B_sel]
    
#     return selected_imgs, processed_queries, original_camera_ids

def process_queries(corrs, sbs_img, num_points=100):
    device = corrs.device
    
    # 입력 데이터가 비어 있는 경우 빈 텐서 반환
    if corrs.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),  # selected_imgs
            torch.empty(0, num_points, 5, device=device),       # processed_queries
            torch.empty(0, device=device)                       # original_camera_ids
        )
    
    # 1. 고유 카메라 인덱스 추출
    unique_cams, counts = torch.unique(corrs[:, 0], 
                                     sorted=False, 
                                     return_counts=True)
    
    # unique_cams가 비어 있는 경우 처리
    if len(unique_cams) == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 5, device=device),
            torch.empty(0, device=device)
        )
    
    # 2. 이미지 선택
    selected_imgs = sbs_img[unique_cams.long()]
    
    # 3. 카메라별 쿼리 처리
    grouped_queries = []
    camera_indices = []
    
    for cam_idx, cam in enumerate(unique_cams):
        mask = (corrs[:, 0] == cam)
        cam_queries = corrs[mask, 1:]  # [N, 4]
        
        # 데이터 트리밍/패딩
        if cam_queries.size(0) >= num_points:
            idx = torch.randperm(cam_queries.size(0))[:num_points]
            selected = cam_queries[idx]
        else:
            padding_size = num_points - cam_queries.size(0)
            selected = torch.cat([
                cam_queries,
                cam_queries[torch.randint(0, cam_queries.size(0), (padding_size,))]
            ], dim=0)
        
        # 카메라 인덱스 추가
        tagged_queries = torch.cat([
            torch.full((num_points, 1), cam_idx, device=device, dtype=corrs.dtype),
            selected
        ], dim=1)
        
        grouped_queries.append(tagged_queries)
        camera_indices.append(cam)
    
    # 4. 배치 차원 생성
    processed_queries = torch.stack(grouped_queries, dim=0)
    original_camera_ids = torch.stack(camera_indices)
    
    return selected_imgs, processed_queries, original_camera_ids

# def process_queries_adv(corrs, sbs_img, num_points=100):
#     device = corrs.device
    
#     # 입력 데이터가 비어 있는 경우 빈 텐서 반환
#     if corrs.numel() == 0:
#         return (
#             torch.empty(0, *sbs_img.shape[1:], device=device),
#             torch.empty(0, num_points, 5, device=device),
#             torch.empty(0, device=device)
#         )
    
#     # 1. 고유 카메라 인덱스 추출
#     unique_cams, counts = torch.unique(corrs[:, 0], sorted=False, return_counts=True)
    
#     # unique_cams가 비어 있는 경우 처리
#     if len(unique_cams) == 0:
#         return (
#             torch.empty(0, *sbs_img.shape[1:], device=device),
#             torch.empty(0, num_points, 5, device=device),
#             torch.empty(0, device=device)
#         )
    
#     # 2. 이미지 선택
#     selected_imgs = sbs_img[unique_cams.long()]
    
#     # 3. 카메라별 쿼리 처리
#     grouped_queries = []
#     camera_indices = []
    
#     for cam_idx, cam in enumerate(unique_cams):
#         mask = (corrs[:, 0] == cam)
#         cam_queries = corrs[mask, 1:]  # [N, 4]
        
#         # 객체별 샘플링 로직 (핵심 수정 부분)
#         if cam_queries.size(0) >= num_points:
#             # 객체 ID 추출 및 분포 계산
#             object_ids = cam_queries[:, 0]
#             unique_objs, obj_counts = torch.unique(object_ids, return_counts=True)
            
#             # 객체 빈도순 정렬 (빈도 높은 객체 우선)
#             sorted_objs = unique_objs[torch.argsort(-obj_counts)]
            
#             # 객체별 최소 1개 샘플링 보장
#             selected_indices = []
#             for obj in sorted_objs:
#                 obj_mask = (object_ids == obj)
#                 obj_indices = torch.where(obj_mask)[0]
#                 if len(selected_indices) < num_points:
#                     idx = torch.randint(0, len(obj_indices), (1,))
#                     selected_indices.append(obj_indices[idx])
            
#             # 남은 슬롯 채우기
#             remaining = num_points - len(selected_indices)
#             if remaining > 0:
#                 all_indices = torch.arange(len(object_ids), device=device)
#                 mask = torch.ones(len(object_ids), dtype=torch.bool, device=device)
                
#                 # Concatenate selected indices into a single tensor
#                 if selected_indices:
#                     selected_indices = torch.cat(selected_indices).to(device)
#                 else:
#                     selected_indices = torch.tensor([], dtype=torch.long, device=device)
                
#                 mask[selected_indices] = False

#                 remaining_indices = all_indices[mask]
#                 if len(remaining_indices) > 0:
#                     idx = torch.randperm(len(remaining_indices))[:remaining]
#                     selected_remaining = remaining_indices[idx]
#                     selected_indices = torch.cat([selected_indices, selected_remaining])
            
#             # 최종 선택 및 충돌 방지
#             selected_indices = torch.tensor(selected_indices[:num_points], device=device)
#             selected = cam_queries[selected_indices]
#         else:
#             # 기존 패딩 로직 (객체 다양성 고려)
#             padding_size = num_points - cam_queries.size(0)
#             selected = torch.cat([
#                 cam_queries,
#                 cam_queries[torch.randint(0, cam_queries.size(0), (padding_size,))]
#             ], dim=0)
        
#         # 카메라 인덱스 추가
#         tagged_queries = torch.cat([
#             torch.full((num_points, 1), cam, device=device, dtype=corrs.dtype),
#             selected
#         ], dim=1)
        
#         grouped_queries.append(tagged_queries)
#         camera_indices.append(cam)
    
#     # 4. 배치 차원 생성
#     processed_queries = torch.stack(grouped_queries, dim=0)
#     original_camera_ids = torch.stack(camera_indices)
    
#     return selected_imgs, processed_queries, original_camera_ids

def process_queries_adv(corrs, sbs_img, num_points=100):
    device = corrs.device
    
    # 입력 데이터가 비어 있는 경우 빈 텐서 반환
    if corrs.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 6, device=device),  # 5 → 6 (confidence 추가)
            torch.empty(0, device=device)
        )
    
    # 1. 고유 카메라 인덱스 추출
    unique_cams, counts = torch.unique(corrs[:, 0], sorted=False, return_counts=True)
    
    # unique_cams가 비어 있는 경우 처리
    if len(unique_cams) == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 6, device=device),  # 5 → 6
            torch.empty(0, device=device)
        )
    
    # 2. 이미지 선택
    selected_imgs = sbs_img[unique_cams.long()]
    
    # 3. 카메라별 쿼리 처리
    grouped_queries = []
    camera_indices = []
    
    for cam_idx, cam in enumerate(unique_cams):
        mask = (corrs[:, 0] == cam)
        cam_queries = corrs[mask, 1:]  # [N, 8] (obj_id, x,y,z, x',y',z', confidence)
        
        # 객체별 샘플링 로직 (핵심 수정 부분)
        if cam_queries.size(0) >= num_points:
            # 객체 ID 추출 및 분포 계산
            object_ids = cam_queries[:, 0]
            unique_objs, obj_counts = torch.unique(object_ids, return_counts=True)
            
            # 객체 빈도순 정렬 (빈도 높은 객체 우선)
            sorted_objs = unique_objs[torch.argsort(-obj_counts)]
            
            # 객체별 최소 1개 샘플링 보장
            selected_indices = []
            for obj in sorted_objs:
                obj_mask = (object_ids == obj)
                obj_indices = torch.where(obj_mask)[0]
                if len(selected_indices) < num_points:
                    idx = torch.randint(0, len(obj_indices), (1,))
                    selected_indices.append(obj_indices[idx])
            
            # 남은 슬롯 채우기
            remaining = num_points - len(selected_indices)
            if remaining > 0:
                all_indices = torch.arange(len(object_ids), device=device)
                mask = torch.ones(len(object_ids), dtype=torch.bool, device=device)
                
                if selected_indices:
                    selected_indices = torch.cat(selected_indices).to(device)
                else:
                    selected_indices = torch.tensor([], dtype=torch.long, device=device)
                
                mask[selected_indices] = False
                remaining_indices = all_indices[mask]
                if len(remaining_indices) > 0:
                    idx = torch.randperm(len(remaining_indices))[:remaining]
                    selected_remaining = remaining_indices[idx]
                    selected_indices = torch.cat([selected_indices, selected_remaining])
            
            # 최종 선택 및 충돌 방지
            selected_indices = selected_indices[:num_points]
            selected = cam_queries[selected_indices]
        # 수정된 else 블록 (패딩 로직)
        else:
            # Confidence 기반 패딩 (중복 허용)
            padding_size = num_points - cam_queries.size(0)
            if padding_size > 0:
                # Confidence 높은 순으로 정렬
                sorted_indices = torch.argsort(cam_queries[:, -1], descending=True)
                # 중복 허용하여 패딩 샘플 선택
                if len(sorted_indices) == 0:
                    padding_samples = cam_queries[torch.randint(0, len(cam_queries), (padding_size,))]
                else:
                    repeat_times = (padding_size // len(sorted_indices)) + 1
                    repeated_indices = sorted_indices.repeat(repeat_times)
                    padding_indices = repeated_indices[:padding_size]
                    padding_samples = cam_queries[padding_indices]
                selected = torch.cat([cam_queries, padding_samples], dim=0)
            else:
                selected = cam_queries
        
        # [추가] Confidence Score 포함하여 쿼리 구성
        tagged_queries = torch.cat([
            torch.full((num_points, 1), cam, device=device, dtype=corrs.dtype),  # cam_id (1)
            selected[:, :11],   # obj_id, x, y, z ,x' ,y', z' (4)
            selected[:, -1].unsqueeze(1)  # confidence (1)
        ], dim=1)  # 총 1+7+1=9 차원
        
        grouped_queries.append(tagged_queries)
        camera_indices.append(cam)
    
    # 4. 배치 차원 생성
    processed_queries = torch.stack(grouped_queries, dim=0)
    original_camera_ids = torch.stack(camera_indices)
    
    return selected_imgs, processed_queries, original_camera_ids


def process_queries_adv_modified(corrs, sbs_img, num_points=100):
    device = corrs.device

    # 0. 입력 데이터 검증 강화
    if corrs.numel() == 0 or corrs.dim() != 2 or corrs.size(1) < 8:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 4, device=device),
            torch.empty(0, device=device)
        )

    # 1. 유효 카메라 인덱스 필터링 (0 ≤ cam_id ≤ max_cam)
    max_cam_id = sbs_img.size(0) - 1
    valid_cam_mask = (corrs[:, 0] >= 0) & (corrs[:, 0] <= max_cam_id)
    corrs = corrs[valid_cam_mask]

    # 2. 고유 카메라 추출 시 차원 검증
    if corrs.size(0) == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 4, device=device),
            torch.empty(0, device=device)
        )

    # 3. 안전한 unique_cams 추출 (long 타입 강제 변환)
    unique_cams = torch.unique(corrs[:, 0].to(torch.long), sorted=False)

    # 4. 이미지 선택 (인덱스 범위 강제 제한)
    selected_imgs = sbs_img[unique_cams.clamp(0, max_cam_id)]

    # 5. 카메라별 처리
    grouped_queries = []
    camera_indices = []

    for cam in unique_cams:
        # 5-1. 마스크 생성 및 차원 검증
        mask = (corrs[:, 0].to(torch.long) == cam)
        if not mask.any():
            continue

        # 5-2. 쿼리 추출 (차원 보호)
        cam_queries = corrs[mask]
        if cam_queries.size(1) < 8:
            continue  # 컬럼 부족 시 스킵

        # 5-3. 객체별 샘플링 (인덱스 범위 검증)
        if cam_queries.size(0) >= num_points:
            object_ids = cam_queries[:, 1]
            unique_objs, obj_counts = torch.unique(object_ids, return_counts=True)
            sorted_objs = unique_objs[torch.argsort(-obj_counts)]

            selected_indices = []
            for obj in sorted_objs:
                obj_mask = (object_ids == obj)
                obj_indices = torch.where(obj_mask)[0]
                if len(selected_indices) < num_points:
                    idx = torch.randint(0, len(obj_indices), (1,), device=device)
                    selected_indices.append(obj_indices[idx])

            # 남은 슬롯 채우기
            remaining = num_points - len(selected_indices)
            if remaining > 0:
                all_indices = torch.arange(len(object_ids), device=device)
                mask = torch.ones(len(object_ids), dtype=torch.bool, device=device)
                selected_indices = torch.cat(selected_indices).to(device)
                mask[selected_indices] = False
                remaining_indices = all_indices[mask]
                if len(remaining_indices) > 0:
                    idx = torch.randperm(len(remaining_indices), device=device)[:remaining]
                    selected_remaining = remaining_indices[idx]
                    selected_indices = torch.cat([selected_indices, selected_remaining])

            # 최종 선택 및 충돌 방지
            selected_indices = selected_indices[:num_points]
            selected = cam_queries[selected_indices]
        else:
            # 5-4. 안전한 패딩 (빈 데이터 방지)
            padding_size = num_points - cam_queries.size(0)
            if padding_size > 0:
                if cam_queries.size(0) == 0:
                    # 빈 쿼리 처리
                    selected = torch.zeros((num_points, 8), device=device, dtype=corrs.dtype)
                else:
                    # 중복 허용 패딩
                    idx = torch.randint(0, cam_queries.size(0), (padding_size,), device=device)
                    padding_samples = cam_queries[idx]
                    selected = torch.cat([cam_queries, padding_samples], dim=0)
            else:
                selected = cam_queries

        # 5-5. 좌표 검증 및 클램핑
        selected_trimmed = selected[:, [2, 3, 6, 7]].clamp(0, 1e5)  # 실제 이미지 크기에 맞게 조정 필요

        # 5-6. 차원 보정 (NaN/Inf 방지)
        selected_trimmed = torch.nan_to_num(selected_trimmed, nan=0.0, posinf=0.0, neginf=0.0)

        # 5-7. 차원 보정 (정확히 num_points로 맞추기)
        if selected_trimmed.size(0) < num_points:
            pad_size = num_points - selected_trimmed.size(0)
            pad_tensor = torch.zeros((pad_size, 4), device=device, dtype=selected_trimmed.dtype)
            selected_trimmed = torch.cat([selected_trimmed, pad_tensor], dim=0)
        elif selected_trimmed.size(0) > num_points:
            selected_trimmed = selected_trimmed[:num_points]

        grouped_queries.append(selected_trimmed)
        camera_indices.append(cam)

    # 6. 배치 차원 생성 (빈 데이터 처리)
    processed_queries = torch.stack(grouped_queries, dim=0) if grouped_queries else torch.empty((0, num_points, 4), device=device)
    original_camera_ids = torch.stack(camera_indices) if camera_indices else torch.empty((0,), device=device)

    return selected_imgs, processed_queries, original_camera_ids


# def differentiable_process_queries(corrs, sbs_img, num_points=100, temp=0.1):
#     device = corrs.device
#     num_cam = 6

#     # 1. 카메라별 그룹화 (One-hot 인코딩)
#     cam_mask = torch.eye(num_cam, device=device)[corrs[:,0].long()]  # [N,6]

#     all_queries = []
#     for cam in range(num_cam):
#         # 2. 카메라별 쿼리 선택
#         cam_corrs = corrs[cam_mask[:,cam].bool()]

#         if len(cam_corrs) == 0:
#             # 빈 카메라 처리
#             all_queries.append(torch.zeros(num_points, 9, device=device))
#             continue

#         # 3. 객체별 확률 계산 (객체가 0개인 경우 처리)
#         obj_ids = cam_corrs[:,1]
#         unique_objs, obj_idx = torch.unique(obj_ids, return_inverse=True)
#         if len(unique_objs) == 0:
#             all_queries.append(torch.zeros(num_points, 9, device=device))
#             continue

#         # 4. 샘플링 개수 조정 (핵심 수정 부분)
#         k_val = min(num_points, len(unique_objs))
        
#         # 5. Gumbel-TopK 샘플링 (k_val 사용)
#         logits = torch.ones(len(unique_objs), device=device)  # 균일 확률
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#         _, selected_obj_idx = torch.topk(logits + gumbel_noise, k_val)

#         # 6. 남은 슬롯 채우기 (중복 허용)
#         if k_val < num_points:
#             repeat_times = (num_points // k_val) + 1
#             selected_obj_idx = selected_obj_idx.repeat(repeat_times)[:num_points]

#         # 7. 객체 내부 랜덤 샘플링
#         sampled_queries = []
#         for obj in selected_obj_idx:
#             mask = (obj_idx == obj)
#             if mask.sum() == 0:  # 예외 처리
#                 sampled_queries.append(cam_corrs[0].unsqueeze(0))
#                 continue
                
#             weights = torch.softmax(torch.randn(mask.sum(), device=device), dim=0)
#             idx = torch.multinomial(weights, 1)
#             sampled_queries.append(cam_corrs[mask][idx])

#         # 8. 결과 조립 및 패딩
#         sampled_queries = torch.cat(sampled_queries)[:num_points]
#         if len(sampled_queries) < num_points:
#             padding = sampled_queries[torch.randint(0, len(sampled_queries), 
#                                    (num_points - len(sampled_queries),))]
#             sampled_queries = torch.cat([sampled_queries, padding])

#         all_queries.append(sampled_queries)

#     # 9. 최종 출력 형식 맞춤
#     # processed_queries = torch.stack([
#     #     torch.cat([q[:,:1], q[:,2:], q[:,1:2]], dim=1) for q in all_queries
#     # ])
#     processed_queries = torch.stack([
#     torch.cat([q[:,:2], q[:,2:]], dim=1) for q in all_queries  # obj_id 위치 수정
#     ])
    
#     return sbs_img, processed_queries, torch.arange(num_cam, device=device)

def differentiable_process_queries(corrs, sbs_img, num_points=100, temp=0.1):
    device = corrs.device
    num_cam = 6

    # 1. 카메라별 그룹화 (One-hot 인코딩)
    cam_mask = torch.eye(num_cam, device=device)[corrs[:,0].long()]  # [N,6]

    all_queries = []
    for cam in range(num_cam):
        # 2. 카메라별 쿼리 선택
        cam_corrs = corrs[cam_mask[:,cam].bool()]
        if len(cam_corrs) == 0:
            # 빈 카메라 처리 (12차원으로 수정)
            all_queries.append(torch.zeros(num_points, 12, device=device))
            continue

        # 3. 객체별 확률 계산 (객체가 0개인 경우 처리)
        obj_ids = cam_corrs[:,1]
        unique_objs, obj_idx = torch.unique(obj_ids, return_inverse=True)
        if len(unique_objs) == 0:
            # 12차원으로 수정
            all_queries.append(torch.zeros(num_points, 12, device=device))
            continue

        # 4. 샘플링 개수 조정 (핵심 수정 부분)
        k_val = min(num_points, len(unique_objs))
        
        # 5. Gumbel-TopK 샘플링 (k_val 사용)
        logits = torch.ones(len(unique_objs), device=device)  # 균일 확률
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        _, selected_obj_idx = torch.topk(logits + gumbel_noise, k_val)

        # 6. 남은 슬롯 채우기 (중복 허용)
        if k_val < num_points:
            repeat_times = (num_points // k_val) + 1
            selected_obj_idx = selected_obj_idx.repeat(repeat_times)[:num_points]

        # 7. 객체 내부 랜덤 샘플링
        sampled_queries = []
        for obj in selected_obj_idx:
            mask = (obj_idx == obj)
            if mask.sum() == 0:  # 예외 처리
                sampled_queries.append(cam_corrs[0].unsqueeze(0))
                continue
                
            weights = torch.softmax(torch.randn(mask.sum(), device=device), dim=0)
            idx = torch.multinomial(weights, 1)
            sampled_queries.append(cam_corrs[mask][idx])

        # 8. 결과 조립 및 패딩
        sampled_queries = torch.cat(sampled_queries)[:num_points]
        if len(sampled_queries) < num_points:
            padding = sampled_queries[torch.randint(0, len(sampled_queries), 
                                   (num_points - len(sampled_queries),))]
            sampled_queries = torch.cat([sampled_queries, padding])

        all_queries.append(sampled_queries)

    # 9. 최종 출력 형식 맞춤 (12차원 유지)
    processed_queries = torch.stack([
        torch.cat([q[:,:2], q[:,2:]], dim=1) for q in all_queries  # [cam_id, obj_id, u, v, z, u', v', z', x', y', z', conf]
    ])
    
    return sbs_img, processed_queries, torch.arange(num_cam, device=device)


def process_queries_adv1(corrs, sbs_img, rois_with_indices, num_points=300):
    device = corrs.device
    
    # 빈 입력 처리
    if corrs.numel() == 0 and rois_with_indices.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 8, device=device),
            torch.empty(0, device=device)
        )

    # 1. ROI 정보에서 (cam_id, obj_id) 추출
    roi_keys = rois_with_indices[:, :2].long().unique(dim=0)
    valid_cams = torch.cat([rois_with_indices[:, 0], corrs[:, 0]]).unique()

    # 유효 카메라 필터링 (0 ≤ cam_id < 6)
    valid_cams = valid_cams[(valid_cams >= 0) & (valid_cams < 6)]

    if valid_cams.numel() == 0:
        return (
            torch.empty(0, *sbs_img.shape[1:], device=device),
            torch.empty(0, num_points, 8, device=device),
            torch.empty(0, device=device)
        )

    # 2. 이미지 선택
    selected_imgs = sbs_img[valid_cams.long()]
    
    grouped_queries = []
    camera_indices = []

    for cam in valid_cams:
        cam = cam.item()
        cam_rois = rois_with_indices[rois_with_indices[:, 0] == cam]
        cam_corrs = corrs[corrs[:, 0] == cam]

        # 3. 현재 카메라의 모든 obj_id 수집
        existing_obj_ids = torch.cat([
            cam_rois[:, 1].long(),
            cam_corrs[:, 1].long()
        ]).unique()
        
        # 4. 유효 obj_id 관리 (0~80)
        valid_obj_ids = existing_obj_ids[(existing_obj_ids >= 0) & (existing_obj_ids < 81)]
        max_obj_id = valid_obj_ids.max().item() if valid_obj_ids.numel() > 0 else 0

        queries = []

        # 5. ROI 우선 처리
        for roi_obj in cam_rois[:, 1].unique():
            roi_obj = roi_obj.item()
            obj_mask = (cam_corrs[:, 1] == roi_obj)
            
            if obj_mask.any():
                # 기존 포인트 사용
                queries.append(cam_corrs[obj_mask][0])
            else:
                # 신규 포인트 생성 (obj_id 범위 검증)
                new_obj_id = roi_obj % 81
                new_point = generate_roi_point(cam, new_obj_id, device)
                queries.append(new_point)

        # 6. 남은 슬롯 채우기
        remaining = num_points - len(queries)
        if remaining > 0:
            # 새로운 obj_id 생성 (순차적 할당 + 모듈로 연산)
            new_obj_ids = (max_obj_id + 1 + torch.arange(remaining, device=device)) % 81
            new_obj_ids = new_obj_ids.cpu().numpy().astype(int)
            
            # 신규 포인트 생성
            for obj_id in new_obj_ids:
                new_point = generate_roi_point(cam, obj_id, device)
                queries.append(new_point)

        # 7. 최종 데이터 포맷팅
        selected = torch.stack(queries[:num_points])
        tagged_queries = torch.cat([
            torch.full((num_points, 1), cam, device=device, dtype=torch.long),
            selected[:, 1:]
        ], dim=1)
        
        grouped_queries.append(tagged_queries)
        camera_indices.append(cam)

    # 8. 출력 형식 변환
    processed_queries = torch.stack(grouped_queries, dim=0) if grouped_queries else torch.empty(0, device=device)
    original_camera_ids = torch.tensor(camera_indices, device=device) if camera_indices else torch.empty(0, device=device)

    return selected_imgs, processed_queries, original_camera_ids

def generate_roi_point(cam_id, obj_id, device, existing_points=None):
    """ROI 기반 신규 포인트 생성 (통계적 분포 반영 버전)"""
    # 텐서 → 스칼라 변환
    if isinstance(cam_id, torch.Tensor):
        cam_id = cam_id.item()
    if isinstance(obj_id, torch.Tensor):
        obj_id = obj_id.item()

    # 기존 포인트가 3개 이상인 경우 통계적 분포 사용
    if existing_points is not None and len(existing_points) >= 3:
        # x,y,z 좌표 추출 (cam_id, obj_id 제외)
        spatial_coords = existing_points[:, 2:5]  # [N,3]
        
        # 평균 및 공분산 계산
        mean = spatial_coords.mean(dim=0)
        cov = torch.cov(spatial_coords.T)
        
        # 수치 안정성을 위한 작은 값 추가
        cov += torch.eye(3, device=device) * 1e-6
        
        # 다변량 정규분포 샘플링
        try:
            mvn = torch.distributions.MultivariateNormal(mean, cov)
            sample = mvn.sample()
            x, y, z = sample[0], sample[1], sample[2]
        except:
            # 특이행렬 경우 대비 (대각 공분산 사용)
            std = torch.sqrt(torch.diag(cov))
            x = torch.normal(mean[0], std[0], (1,))
            y = torch.normal(mean[1], std[1], (1,))
            z = torch.normal(mean[2], std[2], (1,))
    else:
        # 기본 범위 사용 (실제 환경에 맞게 조정 필요)
        x = torch.rand(1, device=device).item() * 100  # 0~100m
        y = torch.rand(1, device=device).item() * 100  # 0~100m
        z = torch.rand(1, device=device).item() * 50   # 0~50m

    return torch.tensor([
        cam_id,
        obj_id,
        x,
        y,
        z,
        0.0,  # x'
        0.0,  # y'
        0.0   # z'
    ], device=device, dtype=torch.float32)

# def trim_corrs_torch(in_corrs, num_kp=100):
#     length = in_corrs.shape[0]
    
#     if length == 0 :
#         reduced_corrs = torch.rand((num_kp, in_corrs.shape[1]), dtype=in_corrs.dtype, device=in_corrs.device) * 1e-6
#     elif in_corrs is None :
#         reduced_corrs =torch.rand((num_kp, in_corrs.shape[1]), dtype=in_corrs.dtype, device=in_corrs.device) * 1e-6
#     elif length >= num_kp:
#         mask = torch.randperm(length)[:num_kp]
#         reduced_corrs =in_corrs[mask]
#     else:
#         mask = torch.randperm(length).repeat(num_kp // length + 1)[:num_kp]
#         reduced_corrs = in_corrs[mask]
    
#     return reduced_corrs 

def trim_corrs_torch(in_corrs, num_kp=100):
    if in_corrs is None or in_corrs.shape[0] == 0:
        # Handle None or empty input
        return torch.rand((num_kp, 6), dtype=torch.float32, device='cuda') * 1e-6

    length = in_corrs.shape[0]
    device = in_corrs.device

    if length >= num_kp:
        mask = torch.randperm(length, device=device)[:num_kp]
    else:
        mask = torch.randperm(length, device=device).repeat(num_kp // length + 1)[:num_kp]

    return in_corrs[mask]


def resize_points(query_xyz):
    original_height, original_width = 900,1600
    target_height, target_width = 192,640

    # 스케일 계산
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    # 점 좌표 조정
    resized_points = query_xyz.clone()
    resized_points[..., 0] *= scale_w  # x 좌표 조정
    resized_points[..., 1] *= scale_h  # y 좌표 조정

    return resized_points

def farthest_point_sampling(points, k):
    """
    Args:
        points (torch.Tensor): (N, 3) shape의 포인트 집합
        k (int): 선택할 중심 포인트의 개수
    Returns:
        torch.Tensor: (k, 3) shape의 선택된 중심 포인트 좌표
        torch.Tensor: (k) shape의 선택된 중심 포인트 인덱스
    """
    N, _ = points.shape
    centroids = torch.zeros(k, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10

    # 첫 번째 중심 포인트를 무작위로 선택
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=points.device)

    for i in range(k):
        # 가장 먼 지점을 중심 포인트로 선택
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)

        # 선택한 중심 포인트와 다른 모든 포인트 간의 거리 계산
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

        # 가장 먼 포인트를 찾는다
        farthest = torch.max(distance, dim=0)[1]

    # 선택된 중심 포인트 좌표 및 인덱스 반환
    return centroids ,points[centroids]

def knn(x, y ,k):
# #         print (" x shape = " , x.shape)
#         inner = -2*torch.matmul(x.transpose(-2, 1), x)
#         xx = torch.sum(x**2, dim=1, keepdim=True)
# #         print (" xx shape = " , x.shape)
#         pairwise_distance = -xx - inner - xx.transpose(4, 1)
    # mask_x = (x[: , 2] > 0.5) & (x[: , 2] < 0.8)
    # mask_y = (y[: , 2] > 0.5) & (y[: , 2] < 0.8)
    # x1 = x[mask_x]
    # y1 = y[mask_y]
    # mask_x1= np.in1d(mask_x,mask_y)
    # mask_y1= np.in1d(mask_y,mask_x)
    # x2 = x[mask_x1]
    # y2 = y[mask_y1]
    # x2 = torch.from_numpy(x2)  # NumPy 배열을 PyTorch Tensor로 변환
    # y2 = torch.from_numpy(y2)  # NumPy 배열을 PyTorch Tensor로 변환
    # pairwise_distance = F.pairwise_distance(x,y)
    
    # #### monitoring x/y point #####################
    # print ("x2 x_point min =" , torch.min(x[:,0]))
    # print ("x2 x_point max =" , torch.max(x[:,0]))
    # print ("y2 x_point min =" , torch.min(y[:,0]))
    # print ("y2 x_point max =" , torch.max(y[:,0]))
    # print ("x2 y_point min =" , torch.min(x[:,1]))
    # print ("x2 y_point max =" , torch.max(x[:,1]))
    # print ("y2 y_point min =" , torch.min(y[:,1]))
    # print ("y2 y_point max =" , torch.max(y[:,1]))
    # print ("x2 depth min =" , torch.min(x[:,2]))
    # print ("x2 depth max =" , torch.max(x[:,2]))
    # print ("y2 depth min =" , torch.min(y[:,2]))
    # print ("y2 depth max =" , torch.max(y[:,2]))
    # ##############################################
    
    # 일정 depth range (min_depth, max_depth)
    min_depth = 0.05
    max_depth = 0.2
    
    # y[:, 2] = 1 - y[:, 2] # 세 번째 열 값 반전
    # min_depth <= depth <= max_depth 인 point들의 인덱스를 구합니다.
    depth_mask1 = (x[:, 2] >= min_depth) & (x[:, 2] <= max_depth) # & (x[:,1] >= 0.6 )
    depth_mask2 = (y[:, 2] >= min_depth) & (y[:, 2] <= max_depth) # & (y[:,1] >= 0.6 )
    # depth_indices1 = np.where(depth_mask1)[0]
    # depth_indices2 = np.where(depth_mask2)[0]
    depth_indices1 = torch.nonzero(depth_mask1).squeeze()
    depth_indices2 = torch.nonzero(depth_mask2).squeeze()

    x1 = x[depth_indices1]
    y1 = y[depth_indices2]

    # mask_x1= np.in1d(depth_indices1,depth_indices2)
    # mask_y1= np.in1d(depth_indices2,depth_indices1)
    mask_x1 = (depth_indices1.view(-1, 1)== depth_indices2.view(1, -1)).any(dim=1)
    mask_y1 = (depth_indices2.view(-1, 1) == depth_indices1.view(1, -1)).any(dim=1)
    # mask_x1 = torch.tensor([elem in depth_indices2.cpu().numpy() for elem in depth_indices1.cpu().numpy()], device=x.device, dtype=torch.bool)
    # mask_y1 = torch.tensor([elem in depth_indices1.cpu().numpy() for elem in depth_indices2.cpu().numpy()], device=y.device, dtype=torch.bool)

    x2 = x1.index_select(0, torch.nonzero(mask_x1).squeeze())
    y2 = y1.index_select(0, torch.nonzero(mask_y1).squeeze())
    # x2 = x1[mask_x1]
    # y2 = y1[mask_y1]
    
    if x2.shape[0] <= k :
        # x2 = torch.zeros(k, 3 , device=x.device)
        # y2 = torch.zeros(k, 3,  device=y.device)
        ### 부족하면 무조건 랜덤 수 채우기
        x2 = torch.rand(k, 3).cuda()
        y2 = torch.rand(k, 3).cuda()
            
  
    #### 유사한 포인트 뽑기 using KNN #####
    pairwise_distance = F.pairwise_distance(x2, y2)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    # top_indices = torch.topk(pairwise_distance.flatten(), k=k, largest=False)
    # top_indices = top_indices.indices
    # indices = np.unravel_index(top_indices, pairwise_distance.shape)
    # top_indices = np.asarray(top_indices).T
    
    #### 가장 먼 포인트 들 뽑기 #########
    # idx ,_ = farthest_point_sampling(x2,k)

    ########## 랜덤으로 포인트 뽑기 #########
    # idx = torch.randperm(x2.shape[0])[:k]

    top_x = x2[idx]
    top_y = y2[idx]
    # print ("x point of z =" , top_x[3])
    # print ("y point of z =" , top_y[3])
    # top_y[:, 2] =  1- top_y[:, 2] # 세 번째 열의 값에서 1을 빼기 
    # print ("y point of rev z =" , top_y[3])
    
    corrs = torch.cat([top_x,top_y] ,dim=1) 
        
    return idx , corrs

def two_images_side_by_side_np(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    h, w, c = img_a.shape
#         b,h, w, c = img_a.shape
    canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, 0 * w:1 * w, :] = img_a
    canvas[:, 1 * w:2 * w, :] = img_b
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
#         canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
#         canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()

    #canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
    #canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas

def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype

    img_a = img_a.permute(0,2,3,1)
    img_b = img_b.permute(0,2,3,1)
    b, h, w, c = img_a.shape
#         canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
#         canvas[:, 0 * w:1 * w, :] = img_a
#         canvas[:, 1 * w:2 * w, :] = img_b
    canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
    canvas[:, :, 0 * w:1 * w, :] = img_a.detach().cpu().numpy()
    # canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()
    canvas[:, :, 1 * w:2 * w, :] = img_b.detach().cpu().numpy()

#         canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
#         canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
    return canvas

def two_images_side_by_side_gpu(img_a, img_b):
    """
    GPU 텐서용 사이드 바이 사이드 이미지 생성
    입력: [B, C, H, W]
    출력: [B, H, 2*W, C]
    """
    # 1. 입력 검증
    assert img_a.shape == img_b.shape, f"Shape mismatch: {img_a.shape} vs {img_b.shape}"
    assert img_a.device == img_b.device, "Device mismatch"
    
    # 2. 차원 재배열 [B, C, H, W] → [B, H, W, C]
    img_a = img_a.permute(0, 2, 3, 1)  # B H W C
    img_b = img_b.permute(0, 2, 3, 1)
    
    # 3. 캔버스 생성 (GPU 유지)
    b, h, w, c = img_a.shape
    canvas = torch.zeros((b, h, 2 * w, c), 
                        dtype=img_a.dtype,
                        device=img_a.device)
    
    # 4. 이미지 배치별 병합 (GPU 연산)
    canvas[:, :, :w, :] = img_a
    canvas[:, :, w:, :] = img_b
    
    return canvas

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts ,n, m, grid):
    ng = 2 * grid + 1

    # mX = np.zeros((m,n)) + np.float("inf")
    # mY = np.zeros((m,n)) + np.float("inf")
    # mD = np.zeros((m,n))

    # mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    # mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    # mD[np.int32(Pts[1]),np.int32(Pts[1])] = Pts[2]

    # KmX = np.zeros((ng, ng, m - ng, n - ng))
    # KmY = np.zeros((ng, ng, m - ng, n - ng))
    # KmD = np.zeros((ng, ng, m - ng, n - ng))

    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device='cuda')
    mD = torch.zeros((m, n), dtype=torch.float32, device='cuda')

    mX_idx = torch.tensor(Pts[1], dtype=torch.int64)
    mY_idx = torch.tensor(Pts[0], dtype=torch.int64)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device='cuda')

    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    # S = np.zeros_like(KmD[0,0])
    # Y = np.zeros_like(KmD[0,0])
    S = torch.zeros_like(KmD[0, 0])
    Y = torch.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            # s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            s = 1 / torch.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i,j]
            S = S + s

    S[S == 0] = 1
    # out = np.zeros((m,n))
    out = torch.zeros((m, n), dtype=torch.float32, device='cuda')
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
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
    colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32).cuda()
    # colormapped_tensor = torch.from_numpy(colormapped_im).
    return colormapped_tensor

# corr dataset generation 
def corr_gen( gt_points_index, points_index, gt_uv, uv , num_kp = 500) :
    
    inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    inter_uv_mask    = np.in1d(points_index , gt_points_index)
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    corrs = np.concatenate([gt_uv, uv], axis=1)
    corrs = torch.tensor(corrs)
    
    ## corrs 384*1280 image(original image shape) normalization
    corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 2] = (0.5*corrs[:, 2])/1280 + 0.5        
    corrs[:, 3] = (0.5*corrs[:, 3])/384   
    
    if corrs.shape[0] <= num_kp :
        corrs = torch.zeros(num_kp, 4)
        corrs[:, 2] = corrs[:, 2] + 0.5

    corrs_knn_idx = knn(corrs[:,:2], corrs[:,2:], num_kp) # knn 2d point-cloud trim
    corrs = corrs[corrs_knn_idx]               

    assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    
    return corrs

def corr_gen_withZ( gt_points_index, points_index, gt_uv, uv , gt_z, z, origin_img_shape, resized_shape, num_kp = 500) :
    
    #only numpy operation
    # inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    # inter_uv_mask    = np.in1d(points_index , gt_points_index)

    inter_gt_uv_mask = torch.tensor(np.in1d(gt_points_index.cpu().numpy(), points_index.cpu().numpy()), device='cuda')
    inter_uv_mask = torch.tensor(np.in1d(points_index.cpu().numpy(), gt_points_index.cpu().numpy()), device='cuda')
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    gt_z = gt_z[inter_gt_uv_mask]
    z    = z[inter_uv_mask] 
    # gt_uvz = np.concatenate([gt_uv,gt_z], axis=1)
    # uvz= np.concatenate([uv,z],axis=1)
    # corrs = np.concatenate([gt_uvz, uvz], axis=1)
    # corrs = torch.tensor(corrs)
    gt_uvz = torch.cat([gt_uv, gt_z], dim=1)
    uvz = torch.cat([uv, z], dim=1)
    corrs = torch.cat([gt_uvz, uvz], dim=1)

    # gt_points = torch.tensor(gt_uvz)
    # target_points = torch.tensor(uvz)
    # scale_img = np.array (resized_shape) / np.array(origin_img_shape) 
    
    # #### monitoring x/y point #####################
    # print ("origin gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("origin gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("origin target x_point min =" , torch.min(corrs[:,3]))
    # print ("origin target x_point max =" , torch.max(corrs[:,3]))
    # print ("origin gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("origin gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("origin target y_point min =" , torch.min(corrs[:,1]))
    # print ("origin target y_point max =" , torch.max(corrs[:,1]))
    # print ("origin gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("origin gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("origin target depth min =" ,   torch.min(corrs[:,2]))
    # print ("origin target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################
    
    # corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 0] = corrs[:, 0]/origin_img_shape[1] 
    # corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 1] = corrs[:, 1]/origin_img_shape[0] 
    if corrs[:, 2].numel() > 0:
        corrs[:, 2] = (corrs[:, 2]-torch.min(corrs[:, 2]))/(torch.max(corrs[:, 2]) - torch.min(corrs[:, 2]))
    else :
        corrs[:, 2] = (corrs[:, 2]-0)/(80 - 0)
    # corrs[:, 3] = (0.5*corrs[:, 3])/1280 + 0.5
    corrs[:, 3] = corrs[:, 3]/origin_img_shape[1]         
    # corrs[:, 4] = (0.5*corrs[:, 4])/384
    corrs[:, 4] = corrs[:, 4]/origin_img_shape[0]
    if corrs[:, 5].numel() > 0:
        corrs[:, 5] = (corrs[:, 5]-torch.min(corrs[:, 5]))/(torch.max(corrs[:, 5]) - torch.min(corrs[:, 5])) 
    else :
        corrs[:, 5] = (corrs[:, 5]-0)/(80 - 0)

    # #### monitoring x/y point #####################
    # print ("normalized gt x_point min =" ,     torch.min(corrs[:,0]))
    # print ("normalized gt x_point max =" ,     torch.max(corrs[:,0]))
    # print ("normalized target x_point min =" , torch.min(corrs[:,3]))
    # print ("normalized target x_point max =" , torch.max(corrs[:,3]))
    # print ("normalized gt y_point min =" ,     torch.min(corrs[:,1]))
    # print ("normalized gt y_point max =" ,     torch.max(corrs[:,1]))
    # print ("normalized target y_point min =" , torch.min(corrs[:,1]))
    # print ("normalized target y_point max =" , torch.max(corrs[:,1]))
    # print ("normalized gt depth min =" ,       torch.min(corrs[:,2]))
    # print ("normalized gt depth max =" ,       torch.max(corrs[:,2]))
    # print ("normalized target depth min =" ,   torch.min(corrs[:,2]))
    # print ("normalized target depth max =" ,   torch.max(corrs[:,2]))
    # ##############################################

    if corrs.shape[0] <= num_kp :
        # corrs = torch.zeros(num_kp, 6)
        diff = num_kp - corrs.shape[0]
        rand_values = torch.randn(diff, 6).cuda()
        corrs = torch.cat([corrs, rand_values], dim=0)
        # target_points = torch.zeros(num_kp, 3)
        # corrs[:, 2] = corrs[:, 2] + 0.5 # for only uv matching
        # corrs[:, 3] = corrs[:, 3] + 0.5 # for uvz matching

    corrs_knn_idx ,corrs_prev = knn(corrs[:,:3], corrs[:,3:], num_kp) # knn 2d point-cloud trim

    corrs = corrs[corrs_knn_idx]   
    corrs1 = corrs_prev
    # corrs = corrs[z_mask]    
    # corrs = torch.cat([top_gt_points,top_target_points],dim=1)

    # assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    # assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    # assert (0.0 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    # assert (0.5 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    # assert (0.0 <= corrs[:, 4]).all() and (corrs[:, 4] <= 1.0).all()
    # assert (0.0 <= corrs[:, 5]).all() and (corrs[:, 5] <= 1.0).all()
    
    return corrs1

def random_mask(sbs_img,grid_size=(32, 32), mask_value=0):
    # sbs_img shape: [batch, channel, height, width]
    batch_size, _, height, width = sbs_img.shape
    mask = torch.ones_like(sbs_img)

    grid_height, grid_width = grid_size

    for i in range(height // grid_height):
        for j in range(width // grid_width):
            if torch.rand(1) > 0.75:  # Randomly choose whether to mask this grid or not
                # Apply the mask to the corresponding area in the image
                mask[:, :, i*grid_height:(i+1)*grid_height,j*grid_width:(j+1)*grid_width] = mask_value

    return sbs_img * mask


def draw_points(img, query , mode='640*192'):

    img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 포인트 그리기
    if mode=='640*192':
        query *= np.array([640,192])
    elif mode=='320_*92':
        query *= np.array([320,192])
    
    for (x, y) in query:
        draw.ellipse((x-1, y-1, x+1, y+1), fill='red', outline='red')
    
    return np.array(img)

def draw_points_torch(img, query, mode='640*192'):
    # 이미지 정규화 및 PIL 이미지로 변환
    img = img.clone()  # 텐서 복사
    img = (img - img.min()) / (img.max() - img.min()) * 255  # 정규화 후 0-255로 스케일링
    img = img.byte().cpu().numpy()  # 텐서를 NumPy로 변환
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)

    # 포인트 스케일링
    if mode == '640*192':
        query = query * torch.tensor([640, 192], device=query.device)
    elif mode == '320*192':
        query = query * torch.tensor([320, 192], device=query.device)

    # 포인트 그리기
    for (x, y) in query.cpu().numpy():
        draw.ellipse((x-1, y-1, x+1, y+1), fill='red', outline='red')

    return torch.tensor(np.array(img))  # 결과를 다시 텐서로 변환


def draw_center_point(self, img):
    
    img = Image.fromarray(np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 이미지 크기 구하기
    w, h = img.size

    # 정중앙 위치 계산하기
    query_x = w // 2
    query_y = h // 2
    radius = 1
    # 포인트 그리기
    draw.ellipse((query_x-radius, query_y-radius, query_x+radius, query_y+radius), fill='red', outline='red')

    return np.array(img)

def draw_corrs(imgs, corrs, col=(255, 0, 0)):
    imgs = utils.torch_img_to_np_img(imgs)
    out = []
    
    # 삭제하려는 열의 인덱스 리스트
    cols_to_remove = [2, 5] # Python은 0부터 시작하기 때문에 
    # 유지하려는 열들만 선택합니다.
    cols_to_keep = [i for i in range(corrs.shape[2]) if i not in cols_to_remove]
    # index_select 함수를 사용하여 해당 열들만 선택합니다.
    corrs_shrink = corrs[:,:,cols_to_keep]

    for img, corr in zip(imgs, corrs_shrink):
        img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
#             corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
        corr *= np.array([1280,192,1280,192])
        for c in corr:
            draw.line(c, fill=col)
        out.append(np.array(img))
    out = np.array(out) / 255.0
    return utils.np_img_to_torch_img(out) , out 

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output 

def find_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device  # depth_map의 디바이스를 기준으로 설정
    batch_size,num_cam,h,w=depth_map.shape
    depth_map_re = depth_map.view(batch_size*num_cam,h,w)

    for cid in range(num_cam):
        # # x 중심점 계산: (x_min + x_max) / 2
        center_x = (detections[cid][:, 0] + detections[cid][:, 2]) / 2
        # y 중심점 계산: (y_min + y_max) / 2
        center_y = (detections[cid][:, 1] + detections[cid][:, 3]) / 2
        # 중심점 좌표를 하나의 텐서로 결합
        center_points = torch.stack((center_x, center_y), dim=1)
        center_points_int = center_points.long()
        z_values = []

        for x, y in center_points_int:
            if 0 <= x < depth_map_re[cid].shape[1] and 0 <= y < depth_map_re[cid].shape[0]:
                z_values.append(depth_map_re[cid][y, x])  # depth map에서 값 추출
            else:
                z_values.append(torch.tensor(float('nan')).to(device))  # 범위를 벗어난 경우 NaN 할당
        
        z_values = torch.stack(z_values).unsqueeze(1) # z 값을 열 벡터로 변환
        concat_result = torch.cat((center_points_int, z_values), dim=1)  # x, y, z 결합
        result_list.append(concat_result)

    return result_list

def find_all_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    for cid in range(num_cam):
        camera_results = []
        for bbox in detections[cid]:
            x_min, y_min, x_max, y_max = bbox[:4].long()
            
            # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)
            
            # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
            bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
            
            # x, y 좌표 생성
            y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1), torch.arange(x_min, x_max+1))
            coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).to(device)
            
            # z값과 좌표 결합
            z_values = bbox_depth.flatten().unsqueeze(1)
            bbox_points = torch.cat((coords, z_values), dim=1)
            
            camera_results.append(bbox_points)
        
        result_list.append(camera_results)

    return result_list

def find_all_depthmap_z_adv(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    all_bbox_points = []

    for bbox in detections:
        cid = int(bbox[0])  # Get camera ID
        x_min, y_min, x_max, y_max = bbox[1:5].long()
        
        # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)
        
        # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
        bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
        
        # x, y 좌표 생성
        y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1), torch.arange(x_min, x_max+1), indexing='ij')
        coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).to(device)
        
        # z값과 좌표 결합
        z_values = bbox_depth.flatten().unsqueeze(1)
        bbox_points = torch.cat((coords, z_values), dim=1)
        
        all_bbox_points.append(bbox_points)

    if all_bbox_points:
        # 모든 포인트를 하나의 텐서로 결합
        combined_points = torch.cat(all_bbox_points, dim=0)
        
        # 중복된 (x, y) 좌표 제거 및 해당하는 z 값 유지
        unique_coords, unique_indices = torch.unique(combined_points[:, :2], dim=0, return_inverse=True)
        unique_z_values = torch.zeros(unique_coords.shape[0], 1, device=device)
        
        # 중복된 좌표의 z 값 평균 계산
        for i in range(unique_coords.shape[0]):
            unique_z_values[i] = combined_points[unique_indices == i, 2].mean()
        
        # 고유한 좌표와 해당하는 z 값을 결합
        return torch.cat((unique_coords, unique_z_values), dim=1)
    else:
        return torch.empty((0, 3), device=device)

def find_nonzero_depthmap_z(detections, depth_map):
    result_list = []
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    for cid in range(num_cam):
        camera_results = []
        for bbox in detections[cid]:
            x_min, y_min, x_max, y_max = bbox[:4].long()
            
            # 바운딩 박스가 이미지 범위를 벗어나지 않도록 조정
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w - 1, x_max)
            y_max = min(h - 1, y_max)
            
            # 바운딩 박스 내의 모든 포인트에 대해 z값 추출
            bbox_depth = depth_map_re[cid, y_min:y_max+1, x_min:x_max+1]
            
            # x, y 좌표 생성
            y_coords, x_coords = torch.meshgrid(torch.arange(y_min, y_max+1 ,device=device), torch.arange(x_min, x_max+1,device=device))
            coords = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1)
            
            # z값이 0이 아닌 포인트만 선택
            nonzero_mask = bbox_depth.flatten() != 0
            nonzero_coords = coords[nonzero_mask]
            nonzero_z = bbox_depth.flatten()[nonzero_mask].unsqueeze(1)
            
            # z값과 좌표 결합
            bbox_points = torch.cat((nonzero_coords, nonzero_z), dim=1)
            
            camera_results.append(bbox_points)
        
        result_list.append(camera_results)

    return result_list

def find_rois_depthmap_z(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        center_x = (x_min[i] + x_max[i]) // 2
        center_y = (y_min[i] + y_max[i]) // 2
        
        center_z = depth_map_re[cid, center_y, center_x]
        
        if center_z != 0:
            result.append(torch.tensor([center_x, center_y, center_z], device=device))
        else:
            result.append(torch.tensor([center_x, center_y, float('nan')], device=device))

    return torch.stack(result)

def find_rois_nonzero_z(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        center_x = (x_min[i] + x_max[i]) // 2
        center_y = (y_min[i] + y_max[i]) // 2
        
        center_z = depth_map_re[cid, center_y, center_x]
        
        if center_z == 0:
            bbox_width = x_max[i] - x_min[i] + 1
            bbox_height = y_max[i] - y_min[i] + 1
            
            x_start = max(0, center_x - bbox_width // 2)
            x_end = min(w, center_x + bbox_width // 2 + 1)
            y_start = max(0, center_y - bbox_height // 2)
            y_end = min(h, center_y + bbox_height // 2 + 1)
            
            surrounding_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            max_z = torch.max(surrounding_area)
            
            if max_z > 0:
                max_z_indices = torch.where(surrounding_area == max_z)
                y_offset, x_offset = max_z_indices[0][0], max_z_indices[1][0]
                center_x = x_start + x_offset
                center_y = y_start + y_offset
                center_z = max_z
            else:
                # center_z = float('nan')
                # center_z = 0.0
                center_z = torch.rand(1, device=device) * 60
        
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))

    return torch.stack(result)

def find_rois_nonzero_z_adv(detections, depth_map): # perplexity 알고리즘
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)

    cam_indices = detections[:, 0].long()
    bboxes = detections[:, 1:]
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w - 1)
    y_min = torch.clamp(y_min, 0, h - 1)
    x_max = torch.clamp(x_max, 0, w - 1)
    y_max = torch.clamp(y_max, 0, h - 1)
    
    result = []
    fallback_counter = 0  # 카운터 초기화
    total_boxes = len(detections)  # 전체 박스 개수
    
    for i in range(len(detections)):
        cid = cam_indices[i]
        
        # 바운딩 박스 전체 영역 추출
        roi_depth = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        
        # 유효한 depth 값들만 선택
        valid_depths = roi_depth[roi_depth > 0]
        
        if len(valid_depths) > 0:
            # 유효한 depth 값들의 통계 계산
            max_z = torch.max(valid_depths)
            median_z = torch.median(valid_depths)
            
            # 최대 depth 위치 찾기
            max_z_pos = torch.where(roi_depth == max_z)
            local_y, local_x = max_z_pos[0][0], max_z_pos[1][0]
            
            # 전역 좌표로 변환
            center_x = x_min[i] + local_x
            center_y = y_min[i] + local_y
            
            # 최대값이 이상치일 수 있으므로, 중앙값과 비교하여 선택
            if max_z > 2.0 * median_z:
                center_z = median_z
            else:
                center_z = max_z
        else:
            # 바운딩 박스를 확장하여 주변 영역까지 탐색
            expand_ratio = 0.2  # 20% 확장
            bbox_w = x_max[i] - x_min[i]
            bbox_h = y_max[i] - y_min[i]
            
            x_start = max(0, x_min[i] - int(bbox_w * expand_ratio))
            x_end = min(w, x_max[i] + int(bbox_w * expand_ratio))
            y_start = max(0, y_min[i] - int(bbox_h * expand_ratio))
            y_end = min(h, y_max[i] + int(bbox_h * expand_ratio))
            
            expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            valid_expanded = expanded_area[expanded_area > 0]
            
            if len(valid_expanded) > 0:
                # 확장 영역에서 발견된 유효한 depth의 중앙값 사용
                center_z = torch.median(valid_expanded)
                # 원본 바운딩 박스의 중심점 사용
                center_x = (x_min[i] + x_max[i]) // 2
                center_y = (y_min[i] + y_max[i]) // 2
            else:
                # 마지막 수단으로 바운딩 박스 중심점과 예상 깊이값 사용
                fallback_counter += 1  # else 구문 실행 시 카운터 증가
                center_x = (x_min[i] + x_max[i]) // 2
                center_y = (y_min[i] + y_max[i]) // 2
                # 이미지 크기에 기반한 예상 깊이값 (가까운 거리 선호)
                center_z = torch.tensor(20.0, device=device)  # 기본 예상 깊이값
        
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))
    
    # # 통계 출력
    # print(f"Total boxes: {total_boxes}, Fallback cases: {fallback_counter}, Ratio: {fallback_counter/total_boxes*100:.2f}%")

    return torch.stack(result)

def find_rois_nonzero_z_adv1(detections, depth_map):
    device = depth_map.device  # GPU 장치 사용 보장
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)  # 이미 float32이므로 형 변환 제거
    
    # 통계값 계산 부분
    cam_means = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    cam_medians = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    
    # 배치 평균 계산
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    # 카메라별 통계값 계산
    for cid in range(batch_size * num_cam):
        cam_depth = depth_map_re[cid]
        non_zero = cam_depth[cam_depth > 0]
        
        if len(non_zero) > 0:
            cam_means[cid] = torch.mean(non_zero)
            cam_medians[cid] = torch.median(non_zero)
        else:
            cam_means[cid] = batch_mean
            cam_medians[cid] = batch_mean

    # Detection 처리
    cam_indices = detections[:, 0].long().to(device)  # GPU로 이동
    bboxes = detections[:, 1:].to(device)  # GPU로 이동
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i]
        original_center_x = (x_min[i] + x_max[i]) // 2
        original_center_y = (y_min[i] + y_max[i]) // 2
        
        # 초기값 설정 (UnboundLocalError 방지)
        center_x, center_y = original_center_x, original_center_y
        center_z = depth_map_re[cid, original_center_y, original_center_x]
        
        if center_z == 0:
            # 2nd try: Full bounding box search
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            if bbox_area.numel() > 0:
                max_z = torch.max(bbox_area)
                if max_z > 0:
                    max_pos = torch.nonzero(bbox_area == max_z).float().mean(dim=0)
                    center_y = y_min[i] + int(max_pos[0])
                    center_x = x_min[i] + int(max_pos[1])
                    center_z = max_z
                    
            if center_z == 0:
                # 3rd try: Expanded search area
                bbox_width = x_max[i] - x_min[i] + 1
                bbox_height = y_max[i] - y_min[i] + 1
                x_start = max(0, original_center_x - bbox_width)
                x_end = min(w, original_center_x + bbox_width + 1)
                y_start = max(0, original_center_y - bbox_height)
                y_end = min(h, original_center_y + bbox_height + 1)
                
                expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
                if expanded_area.numel() > 0:
                    max_z = torch.max(expanded_area)
                    if max_z > 0:
                        max_pos = torch.nonzero(expanded_area == max_z).float().mean(dim=0)
                        center_y = y_start + int(max_pos[0])
                        center_x = x_start + int(max_pos[1])
                        center_z = max_z
                        
        if center_z == 0:  # All attempts failed
            # Use camera median -> camera mean -> batch mean hierarchy
            center_z = cam_medians[cid] if cam_medians[cid] > 0 else cam_means[cid]
            center_x, center_y = original_center_x, original_center_y  # Keep original coordinates
        
        # 결과 텐서 생성 (GPU에서 수행)
        result.append(torch.tensor([cid, center_x, center_y, center_z], device=device))
    
    return torch.stack(result)

def find_rois_nonzero_z_adv2(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 통계값 계산 부분 (벡터화)
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    cam_means = torch.where(torch.sum(non_zero_mask, dim=(1,2)) > 0,
                            torch.sum(depth_map_re, dim=(1,2)) / torch.sum(non_zero_mask, dim=(1,2)),
                            batch_mean)
    
    # 중앙값 계산 (근사값 사용)
    k = torch.sum(non_zero_mask, dim=(1,2)) // 2
    cam_medians = torch.zeros_like(cam_means)
    for i in range(batch_size * num_cam):
        if k[i] > 0:
            cam_medians[i] = torch.kthvalue(depth_map_re[i].reshape(-1), k[i].item())[0]
        else:
            cam_medians[i] = batch_mean

    # Detection 처리 (벡터화)
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # 초기 중심점 깊이값
    center_z = depth_map_re[cam_indices, center_y, center_x]
    
    # 2nd try: Full bounding box search (벡터화)
    bbox_areas = [depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1] for i, cid in enumerate(cam_indices)]
    max_z_bbox = torch.stack([area.max() if area.numel() > 0 else torch.tensor(0.0, device=device) for area in bbox_areas])
    
    max_pos_bbox = []
    for area in bbox_areas:
        if area.numel() > 0:
            flat_area = area.reshape(-1)
            max_idx = flat_area.argmax()
            max_pos = torch.tensor([max_idx // area.size(1), max_idx % area.size(1)], device=device)
        else:
            max_pos = torch.tensor([0, 0], device=device)
        max_pos_bbox.append(max_pos)
    max_pos_bbox = torch.stack(max_pos_bbox)
    
    # 3rd try: Expanded search area (벡터화)
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    x_start = torch.clamp(center_x - bbox_width, 0, w-1)
    x_end = torch.clamp(center_x + bbox_width + 1, 0, w)
    y_start = torch.clamp(center_y - bbox_height, 0, h-1)
    y_end = torch.clamp(center_y + bbox_height + 1, 0, h)
    
    expanded_areas = [depth_map_re[cid, y_start[i]:y_end[i], x_start[i]:x_end[i]] for i, cid in enumerate(cam_indices)]
    max_z_expanded = torch.stack([area.max() if area.numel() > 0 else torch.tensor(0.0, device=device) for area in expanded_areas])
    
    max_pos_expanded = []
    for area in expanded_areas:
        if area.numel() > 0:
            flat_area = area.reshape(-1)
            max_idx = flat_area.argmax()
            max_pos = torch.tensor([max_idx // area.size(1), max_idx % area.size(1)], device=device)
        else:
            max_pos = torch.tensor([0, 0], device=device)
        max_pos_expanded.append(max_pos)
    max_pos_expanded = torch.stack(max_pos_expanded)
    
    # 결과 결정 (벡터화)
    center_z = torch.where(center_z > 0, center_z,
                           torch.where(max_z_bbox > 0, max_z_bbox,
                                       torch.where(max_z_expanded > 0, max_z_expanded,
                                                   torch.where(cam_medians[cam_indices] > 0, cam_medians[cam_indices], cam_means[cam_indices]))))
    
    center_x = torch.where(center_z == depth_map_re[cam_indices, center_y, center_x], center_x,
                           torch.where(center_z == max_z_bbox, x_min + max_pos_bbox[:, 1],
                                       torch.where(center_z == max_z_expanded, x_start + max_pos_expanded[:, 1],
                                                   center_x)))
    
    center_y = torch.where(center_z == depth_map_re[cam_indices, center_y, center_x], center_y,
                           torch.where(center_z == max_z_bbox, y_min + max_pos_bbox[:, 0],
                                       torch.where(center_z == max_z_expanded, y_start + max_pos_expanded[:, 0],
                                                   center_y)))
    
    result = torch.stack([cam_indices.float(), center_x.float(), center_y.float(), center_z], dim=1)
    
    return result

def find_rois_nonzero_z_adv3(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 통계값 초기화
    cam_means = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    cam_medians = torch.zeros(batch_size * num_cam, device=device, dtype=torch.float32)
    
    # 배치 평균 계산
    non_zero_mask = depth_map_re > 0
    batch_mean = torch.mean(depth_map_re[non_zero_mask]) if torch.any(non_zero_mask) else torch.tensor(0.0, device=device)
    
    # 카메라별 통계값 계산
    for cid in range(batch_size * num_cam):
        cam_depth = depth_map_re[cid]
        non_zero = cam_depth[cam_depth > 0]
        if len(non_zero) > 0:
            cam_means[cid] = torch.mean(non_zero)
            cam_medians[cid] = torch.median(non_zero)
        else:
            cam_means[cid] = batch_mean
            cam_medians[cid] = batch_mean

    # Detection 처리
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = detections[:, 1].long().to(device)
    bboxes = detections[:, 2:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    def interpolate_depth(depth_slice, x, y, window_size=3):
        x1, x2 = max(0, x-window_size), min(w, x+window_size+1)
        y1, y2 = max(0, y-window_size), min(h, y+window_size+1)
        window = depth_slice[y1:y2, x1:x2]
        valid_depths = window[window > 0]
        return valid_depths.mean() if len(valid_depths) > 0 else 0
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        bbox_width = x_max[i] - x_min[i] + 1
        bbox_height = y_max[i] - y_min[i] + 1
        
        # 동적 서치 윈도우 크기 계산
        search_radius_x = max(bbox_width // 4, 3)
        search_radius_y = max(bbox_height // 4, 3)
        
        original_center_x = (x_min[i] + x_max[i]) // 2
        original_center_y = (y_min[i] + y_max[i]) // 2
        center_x, center_y = original_center_x, original_center_y
        
        # 1단계: 중심점 깊이값 보간
        center_z = interpolate_depth(depth_map_re[cid], original_center_x, original_center_y)
        confidence_score = 1.0
        
        if center_z == 0:
            # 2단계: 동적 윈도우 기반 bbox 검색
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            if bbox_area.numel() > 0:
                std_val = bbox_area.std()
                if std_val > 0:
                    diff = torch.abs(bbox_area - bbox_area.mean())
                    weight_factor = -diff / std_val
                    depth_weights = torch.exp(weight_factor)
                else:
                    depth_weights = torch.ones_like(bbox_area)
                weighted_depths = bbox_area * depth_weights
                max_z = torch.max(weighted_depths)
                if max_z > 0:
                    max_pos = torch.nonzero(weighted_depths == max_z).float().mean(dim=0)
                    center_y = y_min[i] + int(max_pos[0])
                    center_x = x_min[i] + int(max_pos[1])
                    center_z = max_z.item()
                    confidence_score = 0.7
        
        if center_z == 0:
            # 3단계: 확장된 동적 서치
            x_start = max(0, original_center_x - search_radius_x)
            x_end = min(w, original_center_x + search_radius_x + 1)
            y_start = max(0, original_center_y - search_radius_y)
            y_end = min(h, original_center_y + search_radius_y + 1)
            
            expanded_area = depth_map_re[cid, y_start:y_end, x_start:x_end]
            if expanded_area.numel() > 0:
                valid_depths = expanded_area[expanded_area > 0]
                if len(valid_depths) > 0:
                    center_z = torch.median(valid_depths).item()
                    valid_pos = torch.nonzero(expanded_area > 0).float().mean(dim=0)
                    center_y = y_start + int(valid_pos[0])
                    center_x = x_start + int(valid_pos[1])
                    confidence_score = 0.5
        
        if center_z == 0:
            # 최종 fallback: 통계값 사용
            if cam_medians[cid] > 0:
                center_z = cam_medians[cid].item()
                confidence_score = 0.3
            elif cam_means[cid] > 0:
                center_z = cam_means[cid].item()
                confidence_score = 0.2
            else:
                center_z = batch_mean.item()
                confidence_score = 0.1
            center_x, center_y = original_center_x, original_center_y
        
        point = torch.tensor([
            [cid, oid, float(center_x), float(center_y), float(center_z), confidence_score]
        ], device=device)
        all_points.append(point)
    
    return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 6), device=device)

def find_rois_nonzero_z_adv4(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    confidence_scores = torch.zeros(len(detections), device=device)
    
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        cx = (x_min[i] + x_max[i]) // 2
        cy = (y_min[i] + y_max[i]) // 2
        center_z = 0.0
        conf = 0.0
        
        # 1단계: 3x3 윈도우 내 최대 Z 값 검출
        y_start = max(0, cy-1)
        y_end = min(h, cy+2)
        x_start = max(0, cx-1)
        x_end = min(w, cx+2)
        
        window = depth_map_re[cid, y_start:y_end, x_start:x_end]
        valid_depths = window[window > 0]
        
        if valid_depths.numel() > 0:
            max_z = valid_depths.max()
            max_pos = (window == max_z).nonzero()[0]
            local_y = max_pos[0].item()
            local_x = max_pos[1].item()
            center_z = max_z.item()
            cx = x_start + local_x
            cy = y_start + local_y
            conf = 1.0

        # 2단계: BBox 내 최대 Z 값 검출
        if conf == 0.0:
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            valid_depths = bbox_area[bbox_area > 0]
            
            if valid_depths.numel() > 0:
                max_z = valid_depths.max()
                max_pos = (bbox_area == max_z).nonzero()[0]
                local_y = max_pos[0].item()
                local_x = max_pos[1].item()
                center_z = max_z.item()
                cx = x_min[i] + local_x
                cy = y_min[i] + local_y
                conf = 0.8

        # 3단계: 모델 예측값 사용
        if conf == 0.0:
            cx = int(model_pred_z[i,0].item() * w)
            cy = int(model_pred_z[i,1].item() * h)
            center_z = model_pred_z[i,2].item()
            conf = 0.7

        # 좌표 클램핑
        cx = torch.clamp(torch.tensor(cx), 0, w-1).item()
        cy = torch.clamp(torch.tensor(cy), 0, h-1).item()
        
        result.append(torch.tensor(
            [cid, cx, cy, center_z, conf], 
            device=device
        ))
        confidence_scores[i] = conf
    
    return torch.stack(result), confidence_scores

def find_rois_nonzero_z_adv5(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    bboxes = detections[:, 1:].to(device)
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        points = []
        conf = 0.0
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            # 실제 측정값이 있는 경우
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            # 포인트 생성 [cam_id, x, y, z, confidence]
            points = torch.stack([
                torch.full_like(global_x, cid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
        else:
            # 모델 예측값 사용
            cx = (model_pred_z[i,0] * w).clamp(0, w-1)
            cy = (model_pred_z[i,1] * h).clamp(0, h-1)
            cz = model_pred_z[i,2]
            conf = 0.7
            
            # 단일 포인트 생성
            points = torch.tensor([
                [cid, cx, cy, cz, conf]
            ], device=device)
            
        all_points.append(points)
    
    # 최종 출력 [N_points, 5]
    return torch.cat(all_points, dim=0)

def find_rois_nonzero_z_adv6(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    # 객체 ID 자동 생성 (0부터 순차적 할당)
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = torch.arange(len(detections), device=device).long()  # [NEW] 자동 인덱싱
    bboxes = detections[:, 1:].to(device)  # 인덱스 조정 (객체 ID 컬럼 제외)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    # 좌표 클램핑
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()  # 생성된 객체 ID 사용
        points = []
        conf = 0.0
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            points = torch.stack([
                torch.full_like(global_x, cid),
                torch.full_like(global_x, oid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
        else:
            cx = (model_pred_z[i,0] * w).clamp(0, w-1)
            cy = (model_pred_z[i,1] * h).clamp(0, h-1)
            cz = model_pred_z[i,2]
            conf = 0.7
            
            points = torch.tensor([
                [cid,oid,cx, cy, cz, conf]
            ], device=device)
            
        all_points.append(points)
    
    return torch.cat(all_points, dim=0)

def find_rois_nonzero_z_adv7(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    # obj_indices = torch.arange(len(detections), device=device).long()
    obj_indices = detections[:, 1].long().to(device)  # 객체 ID 사용
    bboxes = detections[:, 2:].to(device)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        nonzero_indices = (bbox_area > 0).nonzero()
        
        if nonzero_indices.size(0) > 0:
            global_x = x_min[i] + nonzero_indices[:, 1].float()
            global_y = y_min[i] + nonzero_indices[:, 0].float()
            z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            conf = 1.0
            
            points = torch.stack([
                torch.full_like(global_x, cid),
                torch.full_like(global_x, oid),
                global_x,
                global_y,
                z_values,
                torch.full_like(global_x, conf)
            ], dim=1)
            
            all_points.append(points)
    
    return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 6), device=device)

def find_rois_nonzero_z_adv8(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = detections[:, 1].long().to(device)
    bboxes = detections[:, 2:].to(device)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        
        # BBox 중심 좌표 계산
        cx = (x_min[i] + x_max[i]) // 2
        cy = (y_min[i] + y_max[i]) // 2
        
        # 중심점 깊이 값 확인
        center_z = depth_map_re[cid, cy, cx]
        conf = 1.0  # 기본 신뢰도
        
        if center_z <= 0:
            # 중심점이 유효하지 않으면 bbox 내에서 z값 탐색
            bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            nonzero_indices = (bbox_area > 0).nonzero()
            
            if nonzero_indices.size(0) > 0:
                # 유효한 z값이 있는 경우 평균 계산
                z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
                center_z = z_values.mean().item()
                conf = 0.7  # 주변부 신뢰도
            else:
                # 유효한 z값이 없는 경우
                center_z = 40
                conf = 0.3  # 신뢰도 0
        
        # 결과 포인트 추가 (항상 중심점 사용)
        point = torch.tensor([
            [cid, oid, float(cx), float(cy), center_z, conf]
        ], device=device)
        all_points.append(point)
    
    return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 6), device=device)

def find_rois_nonzero_z_adv9(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    confidence_scores = torch.zeros(len(detections), device=device, dtype=torch.float32)
    
    # Extract components from detections [cam_id, obj_id, x_min, y_min, x_max, y_max]
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = detections[:, 1].long().to(device)
    bboxes = detections[:, 2:].to(device)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    def interpolate_depth(depth_slice, x, y, window_size=3):
        x1, x2 = max(0, x-window_size), min(w, x+window_size+1)
        y1, y2 = max(0, y-window_size), min(h, y+window_size+1)
        window = depth_slice[y1:y2, x1:x2]
        valid_depths = window[window > 0]
        return valid_depths.mean().item() if valid_depths.numel() > 0 else 0.0
    
    result = []
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        orig_cx = (x_min[i].item() + x_max[i].item()) // 2
        orig_cy = (y_min[i].item() + y_max[i].item()) // 2
        cx, cy = orig_cx, orig_cy
        z = 0.0
        conf = 1.0
        
        # Stage 1: Center interpolation
        z = interpolate_depth(depth_map_re[cid], orig_cx, orig_cy)
        if z > 1e-6:
            result.append(torch.tensor([cid, oid, cx, cy, z, conf], device=device))
            confidence_scores[i] = conf
            continue
            
        # Stage 2: Weighted bbox search
        bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
        if bbox_area.numel() > 0:
            weights = torch.exp(-torch.abs(bbox_area - bbox_area.mean()) / (bbox_area.std() + 1e-6))
            weighted = bbox_area * weights
            max_z = weighted.max().item()
            if max_z > 1e-6:
                max_pos = (weighted == max_z).nonzero().float().mean(dim=0)
                cy = y_min[i].item() + int(max_pos[0].item())
                cx = x_min[i].item() + int(max_pos[1].item())
                z = max_z
                conf = 0.8
                result.append(torch.tensor([cid, oid, cx, cy, z, conf], device=device))
                confidence_scores[i] = conf
                continue
                
        # Stage 3: Expanded search
        search_x = max(bbox_area.shape[1]//4, 3)
        search_y = max(bbox_area.shape[0]//4, 3)
        x1 = max(0, orig_cx-search_x)
        x2 = min(w, orig_cx+search_x+1)
        y1 = max(0, orig_cy-search_y)
        y2 = min(h, orig_cy+search_y+1)
        
        expanded = depth_map_re[cid, y1:y2, x1:x2]
        valid = expanded[expanded > 0]
        if valid.numel() > 0:
            z = valid.median().item()
            avg_pos = (expanded > 0).nonzero().float().mean(dim=0)
            cy = y1 + int(avg_pos[0].item())
            cx = x1 + int(avg_pos[1].item())
            conf = 0.7
            result.append(torch.tensor([cid, oid, cx, cy, z, conf], device=device))
            confidence_scores[i] = conf
            continue
            
        # Stage 4: Model prediction
        cx = int(model_pred_z[i,0].item() * w)
        cy = int(model_pred_z[i,1].item() * h)
        z = model_pred_z[i,2].item()
        conf = 0.5
        cx = max(0, min(w-1, cx))
        cy = max(0, min(h-1, cy))
        
        result.append(torch.tensor([cid, oid, cx, cy, z, conf], device=device))
        confidence_scores[i] = conf
    
    return torch.stack(result) if result else torch.empty((0, 6), device=device), confidence_scores

def find_rois_nonzero_z_adv10(detections, depth_map, model_pred_z):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map = depth_map.view(batch_size * num_cam, h, w)
    
    # 데이터 추출
    cam_ids = detections[:, 0].long().to(device)
    obj_ids = detections[:, 1].long().to(device)
    bboxes = detections[:, 2:].clamp(0, torch.tensor([w-1, h-1, w-1, h-1], device=device))
    
    # 중심점 계산 (정수 → 실수)
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    
    # 깊이 맵에서 값 추출 (바일리니어 보간)
    depth_vals = F.grid_sample(
        depth_map[cam_ids].unsqueeze(1),
        torch.stack([cx/(w-1)*2-1, cy/(h-1)*2-1], -1).unsqueeze(1),
        mode='bilinear', align_corners=False
    ).squeeze()

    # 모델 예측 좌표 (정규화 해제)
    pred_cx = model_pred_z[:, 0] * (w-1)
    pred_cy = model_pred_z[:, 1] * (h-1)
    pred_z = model_pred_z[:, 2]
    
    # 조건 마스크
    invalid_mask = (depth_vals <= 0) | torch.isnan(depth_vals)
    final_cx = torch.where(invalid_mask, pred_cx, cx)
    final_cy = torch.where(invalid_mask, pred_cy, cy)
    final_z = torch.where(invalid_mask, pred_z, depth_vals)
    
    # 신뢰도 계산 (확장 가능)
    conf = torch.where(invalid_mask, 0.7, 1.0)
    
    return torch.stack([cam_ids, obj_ids, final_cx, final_cy, final_z, conf], dim=1)

def find_rois_nonzero_z_adv11(detections, depth_map):
    device = depth_map.device
    batch_size, num_cam, h, w = depth_map.shape
    depth_map_re = depth_map.view(batch_size * num_cam, h, w)
    
    cam_indices = detections[:, 0].long().to(device)
    obj_indices = detections[:, 1].long().to(device)
    bboxes = detections[:, 2:].to(device)
    
    x_min, y_min, x_max, y_max = bboxes.long().t()
    
    x_min = torch.clamp(x_min, 0, w-1)
    y_min = torch.clamp(y_min, 0, h-1)
    x_max = torch.clamp(x_max, 0, w-1)
    y_max = torch.clamp(y_max, 0, h-1)
    
    all_points = []
    
    for i in range(len(detections)):
        cid = cam_indices[i].item()
        oid = obj_indices[i].item()
        
        # BBox 중심 좌표 계산
        cx = (x_min[i] + x_max[i]) // 2
        cy = (y_min[i] + y_max[i]) // 2
        
        # 중심점 깊이 값 확인
        center_z = depth_map_re[cid, cy, cx]
        conf = 1.0  # 기본 신뢰도
        
        if center_z <= 0:
            # # 중심점이 유효하지 않으면 bbox 내에서 z값 탐색
            # bbox_area = depth_map_re[cid, y_min[i]:y_max[i]+1, x_min[i]:x_max[i]+1]
            # nonzero_indices = (bbox_area > 0).nonzero()
            
            # if nonzero_indices.size(0) > 0:
            #     # 유효한 z값이 있는 경우 평균 계산
            #     z_values = bbox_area[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            #     center_z = z_values.mean().item()
            #     conf = 0.7  # 주변부 신뢰도
            # else:
            # 유효한 z값이 없는 경우
            center_z = 0.0
            conf = 0.3  # 신뢰도 0
        
        # 결과 포인트 추가 (항상 중심점 사용)
        point = torch.tensor([
            [cid, oid, float(cx), float(cy), center_z, conf]
        ], device=device)
        all_points.append(point)
    
    return torch.cat(all_points, dim=0) if all_points else torch.empty((0, 6), device=device)

def differentiable_find_rois(detections, depth_map, model_pred_z):
    device = depth_map.device
    B, C, H, W = depth_map.shape
    
    # 박스 파라미터 추출 및 정규화
    cam_ids = detections[:, 0].long()  # [N]
    obj_ids = detections[:, 1].long()  # [N]
    boxes = detections[:, 2:]  # [N,4] (x_min, y_min, x_max, y_max)
    
    # 중심점 및 크기 계산 (미분 가능)
    centers = torch.stack([
        (boxes[:,0] + boxes[:,2]) / 2 / (W-1),  # cx [0,1]
        (boxes[:,1] + boxes[:,3]) / 2 / (H-1)   # cy [0,1]
    ], dim=1)  # [N,2]
    
    sizes = torch.stack([
        (boxes[:,2] - boxes[:,0]) / (W-1),  # width [0,1]
        (boxes[:,3] - boxes[:,1]) / (H-1)   # height [0,1]
    ], dim=1)  # [N,2]

    # 샘플링 그리드 생성 (미분 가능)
    grid_size = 5
    x = torch.linspace(-0.5, 0.5, grid_size, device=device)  # [-0.5, 0.5]
    y = torch.linspace(-0.5, 0.5, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [5,5]
    
    # 박스 크기 기반 그리드 스케일링
    grid_x = grid_x[None] * sizes[:, 0, None, None]  # [N,5,5]
    grid_y = grid_y[None] * sizes[:, 1, None, None]  # [N,5,5]
    
    # 중심점 기반 그리드 이동
    grid_x = grid_x + centers[:, 0, None, None]  # [N,5,5]
    grid_y = grid_y + centers[:, 1, None, None]  # [N,5,5]
    
    # 최종 그리드 형성 [N,5,5,2]
    grid = torch.stack([grid_x, grid_y], dim=-1) * 2 - 1  # [-1,1] 범위로 정규화

    # 깊이 맵 샘플링 (미분 가능)
    sampled_depth = F.grid_sample(
        depth_map.repeat_interleave(C, dim=0)[cam_ids],  # [N,C,H,W]
        grid,  # [N,5,5,2]
        mode='bilinear',
        align_corners=True
    )  # [N,1,5,5]

    # 신뢰도 계산
    valid_mask = (sampled_depth > 0).float()  # [N,1,5,5]
    valid_count = valid_mask.sum(dim=[2,3])  # [N,1]
    
    # 가중 평균 깊이 계산
    depth_weights = F.softmax(sampled_depth * valid_mask * 10, dim=-1)
    weighted_depth = (sampled_depth * depth_weights).sum(dim=[2,3])  # [N,1]
    
    # 모델 예측값과 결합
    final_depth = torch.sigmoid(valid_count) * weighted_depth + \
                (1 - torch.sigmoid(valid_count)) * model_pred_z[:, 2:]
    
    # 최종 출력 형성
    results = torch.cat([
        cam_ids.unsqueeze(1).float(),
        obj_ids.unsqueeze(1).float(),
        model_pred_z[:, :2] * torch.tensor([W-1, H-1], device=device),
        final_depth,
        torch.sigmoid(valid_count)
    ], dim=1)

    return results, torch.sigmoid(valid_count).squeeze()

def differentiable_find_rois1(detections, depth_map, eps=1e-6, temp=0.1, chunk_size=64):
    device = depth_map.device
    B, num_cam, H, W = depth_map.shape
    N = detections.shape[0]

    # 좌표 초기화
    cam_ids = detections[:, 0].long()
    x_min, y_min, x_max, y_max = detections[:, 2:6].unbind(1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    output_buffer = []

    # Straight-Through Estimator (STE) 함수 정의
    def round_ste(x):
        return (x.round() - x).detach() + x

    for i in range(0, N, chunk_size):
        chunk_end = min(i + chunk_size, N)
        chunk_idx = slice(i, chunk_end)
        
        # 현재 청크 데이터
        chunk_det = detections[chunk_idx]
        chunk_cam = cam_ids[chunk_idx]
        
        # STE를 이용한 정수 좌표 변환
        chunk_xmin = round_ste(x_min[chunk_idx])
        chunk_ymin = round_ste(y_min[chunk_idx])
        chunk_xmax = round_ste(x_max[chunk_idx])
        chunk_ymax = round_ste(y_max[chunk_idx])

        # 1. 중심점 좌표 계산 (STE 적용)
        cx_int = round_ste((chunk_xmin + chunk_xmax) / 2)
        cy_int = round_ste((chunk_ymin + chunk_ymax) / 2)
        
        # 2. Grid 생성 (Bilinear sampling용)
        grid_x = (cx_int + 0.5) / W * 2 - 1
        grid_y = (cy_int + 0.5) / H * 2 - 1
        grid = torch.stack([grid_x, grid_y], -1).unsqueeze(1).unsqueeze(2)

        # 3. 깊이 샘플링 (Bilinear + STE)
        depth_selected = depth_map[0, chunk_cam].unsqueeze(1)  # [Chunk, 1, H, W]
        
        # Bilinear 샘플링 후 STE로 Nearest 효과 구현
        sampled = F.grid_sample(
            depth_selected,
            grid,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        center_z = (sampled.round() - sampled).detach() + sampled  # STE 적용

        # 4. BBox 마스크 생성 (미분 가능한 방식)
        x_range = torch.arange(W, device=device).float()
        y_range = torch.arange(H, device=device).float()
        
        x_mask = (x_range[None] >= chunk_xmin[:, None]) & (x_range[None] <= chunk_xmax[:, None])
        y_mask = (y_range[None] >= chunk_ymin[:, None]) & (y_range[None] <= chunk_ymax[:, None])
        bbox_mask = x_mask[:, None, :] & y_mask[:, :, None]  # [Chunk, H, W]

        # 5. 주변부 깊이 계산
        depth_valid = (depth_selected.squeeze(1) > eps).float()
        valid_pixels = depth_valid * bbox_mask
        
        weighted_depth = (depth_selected.squeeze(1) * valid_pixels).sum((1,2)) / \
                        (valid_pixels.sum((1,2)) + eps)

        # 6. Confidence 계산
        center_valid = (center_z > eps).float()
        has_valid_pixels = (valid_pixels.sum((1,2)) > 0).float()
        conf = 1.0 * center_valid + 0.7 * (has_valid_pixels - center_valid)

        # 최종 결과 생성
        final_z = torch.where(center_z > eps, center_z, weighted_depth)
        
        chunk_result = torch.stack([
            chunk_det[:, 0].float(),
            chunk_det[:, 1].float(),
            cx[chunk_idx],
            cy[chunk_idx],
            final_z,
            conf
        ], dim=1)
        
        output_buffer.append(chunk_result)

    return torch.cat(output_buffer, dim=0)


def image_to_lidar_global_modi(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 1:]
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            list_xyz_global.append(xyz_global)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
    else:
        xyz_global_torch = torch.empty(0, 3, device=det_uvz.device)
    
    return xyz_global_torch

def image_to_lidar_global_modi(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_indices = []  # New list to store indices
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 1:]
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            list_xyz_global.append(xyz_global)
            list_indices.append(det_uvz[mask, 0])  # Store corresponding indices
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)  # Combine indices and add a dimension
        xyz_global_torch = torch.cat([indices_torch, xyz_global_torch], dim=1)  # Concatenate indices with xyz_global
    else:
        xyz_global_torch = torch.empty(0, 4, device=det_uvz.device)  # Adjust shape to [0, 4]
    
    return xyz_global_torch

def image_to_lidar_global_modi1(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_indices = []
    list_confidence = []  # New list for confidence scores
    list_object_indices =[]
    
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 2:5]  # Only take x,y,z coordinates (excluding confidence)
            confidence = det_uvz[mask, 5]  # Get confidence scores
            ob_id = det_uvz[mask,1]
            id = det_uvz[mask,0]
            
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            
            list_xyz_global.append(xyz_global)
            list_indices.append(id)
            list_object_indices.append(ob_id)
            list_confidence.append(confidence)  # Store confidence scores
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)  # Add confidence scores
        ob_id_torch = torch.cat(list_object_indices, dim=0).unsqueeze(1)
        
        # Concatenate indices, xyz_global, and confidence scores
        xyz_global_torch = torch.cat([indices_torch, ob_id_torch, xyz_global_torch, confidence_torch], dim=1)
    else:
        xyz_global_torch = torch.empty(0, 6, device=det_uvz.device)  # Adjust shape to [0, 5]
    
    return xyz_global_torch

def image_to_lidar_global_modi2(
    det_uvz: torch.Tensor, 
    gt_KT: torch.Tensor,
    original_camera_ids: torch.Tensor  # [K,] 실제 사용할 카메라 ID 텐서 (예: [0, 1, 3, 5])
) -> torch.Tensor:
    """
    det_uvz: [B, N, 5] (B=전체 카메라 수)
    gt_KT: [B, 4, 4]
    original_camera_ids: [K,] 실제 처리할 카메라 ID (0-based 인덱스)
    """
    device = det_uvz.device
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_cam_ids = []
    list_obj_ids = []
    
    # 실제 처리할 카메라 ID 추출
    camera_indices = original_camera_ids.int().tolist()
    
    for cid in camera_indices:
        # 유효한 카메라 인덱스인지 확인
        if cid >= det_uvz.size(0) or cid >= gt_KT.size(0):
            continue
            
        # 현재 카메라 데이터 추출
        img2lidar = inverse_gt_kt[cid]
        camera_det = det_uvz[cid]  # [N, 5]
        
        # 유효한 탐지 필터링 (z > 0)
        valid_mask = camera_det[:, 4] > 1e-6
        if not valid_mask.any():
            continue
            
        # 데이터 추출
        uvz = camera_det[valid_mask, 2:5]  # [M, 3]
        obj_ids = camera_det[valid_mask, 1]  # [M]
        
        # 좌표 변환
        normalize_uvz = torch.cat([
            uvz[:, :2] * uvz[:, 2:3],  # u*z, v*z
            uvz[:, 2:3]
        ], dim=1)
        
        # 동차 좌표 변환
        uvz_homo = torch.cat([
            normalize_uvz,
            torch.ones_like(normalize_uvz[:, :1])
        ], dim=1)
        
        # 3D 변환
        xyz_global = (uvz_homo[:, :3] @ img2lidar[:3, :3].T) + img2lidar[:3, 3]
        
        # 결과 저장
        list_xyz_global.append(xyz_global)
        list_cam_ids.append(torch.full((xyz_global.size(0),), cid, device=device))
        list_obj_ids.append(obj_ids)
    
    # 결과 병합
    if list_xyz_global:
        xyz_global = torch.cat(list_xyz_global, dim=0)
        cam_ids = torch.cat(list_cam_ids, dim=0).unsqueeze(1)
        obj_ids = torch.cat(list_obj_ids, dim=0).unsqueeze(1)
        return torch.cat([cam_ids, obj_ids, xyz_global], dim=1)
    else:
        return torch.empty((0, 5), device=device)
    
def image_to_lidar_global_modi3(det_uvz, gt_KT):
    inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    list_indices = []
    # list_confidence = []  # New list for confidence scores
    list_object_indices =[]
    
    for cid in range(6):
        img2lidar = inverse_gt_kt[cid]
        mask = (det_uvz[:, 0] == cid)
        if mask.any():
            detection_uvz = det_uvz[mask, 2:5]  # Only take x,y,z coordinates (excluding confidence)
            # confidence = det_uvz[mask, 5]  # Get confidence scores
            ob_id = det_uvz[mask,1]
            id = det_uvz[mask,0]
            
            normalize_uvz = torch.cat([detection_uvz[:, :2] * detection_uvz[:, 2:3], detection_uvz[:, 2:3]], dim=1).float()
            uvz_homogeneous = torch.cat([normalize_uvz, normalize_uvz.new_ones([normalize_uvz.shape[0], 1])], dim=1)
            xyz_global = torch.matmul(img2lidar[:3, :3], uvz_homogeneous[:, :3].T).T + img2lidar[:3, 3]
            
            list_xyz_global.append(xyz_global)
            list_indices.append(id)
            list_object_indices.append(ob_id)
            # list_confidence.append(confidence)  # Store confidence scores
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)
        # confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)  # Add confidence scores
        ob_id_torch = torch.cat(list_object_indices, dim=0).unsqueeze(1)
        
        # Concatenate indices, xyz_global, and confidence scores
        xyz_global_torch = torch.cat([indices_torch, ob_id_torch, xyz_global_torch], dim=1)
    else:
        xyz_global_torch = torch.empty(0, 5, device=det_uvz.device)  # Adjust shape to [0, 5]
    
    return xyz_global_torch


# def lidar_to_image_with_index(det_xyz, gt_KT, img_shape=(900,1600)):
#     """
#     Args:
#         det_xyz: [N, 4] (camera_index, x, y, z)
#         gt_KT: [6, 4, 4] (카메라별 변환 행렬)
#         img_shape: (H, W) 이미지 해상도 (세로, 가로)
    
#     Returns:
#         uvz_global_torch: [M, 4] (유효한 포인트만 포함)
#         mask_valid_global: [N] (전체 포인트에 대한 유효성 마스크)
#     """
#     list_uvz_global = []
#     list_indices = []
#     mask_valid_global = torch.zeros(det_xyz.shape[0], dtype=torch.bool, device=det_xyz.device)

#     for cid in range(6):
#         lidar2img = gt_KT[cid]
#         mask = (det_xyz[:, 0] == cid)  # 현재 카메라 ID에 해당하는 포인트 필터링
#         if mask.any():
#             detection_xyz = det_xyz[mask, 1:]  # x,y,z 좌표 추출
#             points_lidar2img = (lidar2img @ detection_xyz.T).T  # [N,3]
#             points_lidar2img = torch.cat([points_lidar2img[:, :2] / points_lidar2img[:, 2:3], points_lidar2img[:, 2:3]], dim=1)  # [N,3] (u,v,z)

#             # 유효한 포인트 필터링
#             pcl_uv = points_lidar2img[:, :2]  # u,v 좌표
#             pcl_z = points_lidar2img[:, 2]   # 깊이 값
#             mask_valid = (
#                 (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) &  # u 범위 체크 (0 < u < W)
#                 (pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) &  # v 범위 체크 (0 < v < H)
#                 (pcl_z > 0)                                           # 깊이 값이 양수인지 확인
#             )
            
#             # 유효한 포인트만 저장
#             valid_points = points_lidar2img[mask_valid]
#             valid_indices = det_xyz[mask][mask_valid][:, 0]
#             # valid_points = points_lidar2img
#             # valid_indices = det_xyz[mask][:, 0]

#             list_uvz_global.append(valid_points)
#             list_indices.append(valid_indices)

#             # 전체 마스크 업데이트
#             mask_valid_global[mask.nonzero(as_tuple=True)[0]] = mask_valid

#     if list_uvz_global:
#         uvz_global_torch = torch.cat(list_uvz_global, dim=0)  # 유효한 포인트 병합
#         indices_torch = torch.cat(list_indices, dim=0).unsqueeze(1)  # 인덱스 병합
        
#         # 인덱스와 uvz 좌표 결합
#         uvz_global_torch = torch.cat([indices_torch, uvz_global_torch], dim=1)  # [M,4]
#     else:
#         uvz_global_torch = torch.empty(0, 4, device=det_xyz.device)  # 빈 텐서 반환
    
#     return uvz_global_torch, mask_valid_global

def lidar_to_image_with_index(det_xyz, gt_KT, img_shape=(900, 1600)):
    """
    Args:
        det_xyz: [N, 4] (camera_index, x, y, z)
        gt_KT: [6, 4, 4] (카메라별 변환 행렬)
        img_shape: (H, W) 이미지 해상도 (세로, 가로)
    
    Returns:
        uvz_global_torch: [M, 4] (유효한 포인트만 포함)
        mask_valid_global: [N] (전체 포인트에 대한 유효성 마스크)
    """
    list_uvz_global = []
    mask_valid_global = torch.zeros(det_xyz.shape[0], dtype=torch.bool, device=det_xyz.device)

    for cid in range(6):
        lidar2img = gt_KT[cid]
        mask = (det_xyz[:, 0] == cid)
        if not mask.any():
            continue

        # [STEP 1] LiDAR 좌표 추출 (x,y,z) 및 homogeneous 좌표 추가
        detection_xyz = det_xyz[mask, 1:4]  # [M,3] (x,y,z)
        ones = torch.ones_like(detection_xyz[:, :1])  # [M,1]
        detection_xyz_h = torch.cat([detection_xyz, ones], dim=1)  # [M,4]

        # [STEP 2] 투영 변환 (4x4 @ 4xM → 4xM → Transpose → [M,4])
        points_cam = (lidar2img @ detection_xyz_h.T).T  # [M,4]

        # [STEP 3] z>0 필터링 (카메라 앞쪽 점만 유효)
        z = points_cam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            continue

        # [STEP 4] UV 좌표 계산 (x/z, y/z)
        points_cam_valid = points_cam[valid_z]
        uv = points_cam_valid[:, :2] / points_cam_valid[:, 2:3]  # [K,2]
        uvz = torch.cat([uv, points_cam_valid[:, 2:3]], dim=1)  # [K,3]

        # [STEP 5] 이미지 경계 내 UV 확인
        H, W = img_shape
        mask_uv = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
        )
        uvz_valid = uvz[mask_uv]  # [L,3]

        # [STEP 6] 유효한 인덱스 매핑
        mask_combined = valid_z.clone()
        mask_combined[valid_z] = mask_uv  # z>0 중 UV 유효한 점
        original_indices = mask.nonzero(as_tuple=True)[0][valid_z][mask_uv]
        mask_valid_global[original_indices] = True

        # [STEP 7] 결과 저장 (camera_index 추가)
        indices_torch = torch.full((uvz_valid.shape[0], 1), cid, device=det_xyz.device)
        list_uvz_global.append(torch.cat([indices_torch, uvz_valid], dim=1))

    return (
        torch.cat(list_uvz_global, dim=0) if list_uvz_global 
        else torch.empty((0, 4), device=det_xyz.device),
        mask_valid_global
    )

def diff_lidar_to_image_with_index(det_xyz, gt_KT, img_shape=(900, 1600), temp=100):
    """
    미분 가능한 LiDAR → 이미지 투영 (빈 출력 문제 해결)
    Args:
        det_xyz: [N,4] (cam_id, x,y,z)
        gt_KT: [6,4,4] 카메라 변환 행렬
        img_shape: (H,W) 이미지 해상도
        temp: 소프트 마스크 온도 (기본 100)
    """
    device = det_xyz.device
    H, W = img_shape
    num_cam = gt_KT.shape[0]  # 6

    # 1. 카메라 인덱스 정수화 (STE 적용)
    cam_indices = torch.clamp(det_xyz[:,0].round(), 0, num_cam-1).long()

    # 2. 각 포인트에 해당하는 카메라 행렬 선택
    lidar2img = gt_KT[cam_indices]  # [N,4,4]

    # 3. 투영 변환
    xyz = det_xyz[:,1:4]
    ones = torch.ones_like(xyz[:,:1])
    xyz_h = torch.cat([xyz, ones], dim=1).unsqueeze(-1)  # [N,4,1]
    points_cam = (lidar2img @ xyz_h).squeeze(-1)  # [N,4]

    # 4. UV 좌표 계산 (원본과 동일)
    z = points_cam[:,2] + 1e-6
    uv = points_cam[:,:2] / z.unsqueeze(-1)  # [N,2]

    # 5. 소프트 마스킹 (미분 가능)
    valid_z = torch.sigmoid(temp*(z - 1e-6))  # z>0 근사
    uv_x = (uv[:,0] + 1e-6).clamp(0, W-1)  # [0,W) 강제 클램핑
    uv_y = (uv[:,1] + 1e-6).clamp(0, H-1)  # [0,H) 강제 클램핑
    mask = valid_z  # z 유효성만 고려

    # 6. 최종 출력 구성 (원본과 동일 형식)
    uvz_global = torch.stack([
        cam_indices.float(),
        uv_x,
        uv_y,
        z
    ], dim=1)

    # 7. 하드 마스크 적용 (출력 일치 보장)
    with torch.no_grad():
        hard_mask = (z > 1e-6) & (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
        uvz_global = uvz_global[hard_mask]
    
    return uvz_global, hard_mask


def lidar_to_image_no_filter(det_xyz, gt_KT):
    """
    Args:
        det_xyz: [N, 4] (camera_index, x, y, z)
        gt_KT: [6, 4, 4] (카메라별 변환 행렬)
    
    Returns:
        uvz_global_torch: [M, 4] (모든 포인트 포함)
    """
    list_uvz_global = []
    list_indices = []

    for cid in range(6):
        lidar2img = gt_KT[cid]
        mask = (det_xyz[:, 0] == cid)  # 현재 카메라 ID에 해당하는 포인트 필터링
        if mask.any():
            # [STEP 1] Homogeneous 좌표 추가 (x,y,z → x,y,z,1)
            detection_xyz = det_xyz[mask, 1:4]  # x,y,z 좌표 추출
            ones = torch.ones_like(detection_xyz[:, :1])  # [N,1]
            detection_xyz_h = torch.cat([detection_xyz, ones], dim=1)  # [N,4]
            
            # [STEP 2] 올바른 행렬 곱셈 (4x4 @ 4xN → 4xN → Transpose → [N,4])
            points_lidar2img = (lidar2img @ detection_xyz_h.T).T  # [N,4]

            # [STEP 3] z 값 필터링 (z > 1e-6)
            z = points_lidar2img[:, 2]
            valid_mask = z > 1e-6
            if valid_mask.any():
                points_valid = points_lidar2img[valid_mask]
                u = points_valid[:, 0] / points_valid[:, 2]
                v = points_valid[:, 1] / points_valid[:, 2]
                uvz = torch.stack([u, v, points_valid[:, 2]], dim=1)  # [M,3]
                
                list_uvz_global.append(uvz)
                list_indices.append(det_xyz[mask][valid_mask][:, 0].unsqueeze(1))

    if list_uvz_global:
        uvz_global = torch.cat(list_uvz_global, dim=0)
        indices = torch.cat(list_indices, dim=0)
        uvz_global_torch = torch.cat([indices, uvz_global], dim=1)  # [M,4]
    else:
        uvz_global_torch = torch.empty((0,4), device=det_xyz.device)
    
    return uvz_global_torch

def project_lidar_to_image(pts_hom: torch.Tensor, 
                           lidar2img: torch.Tensor, 
                           eps: float = 1e-6) -> torch.Tensor:
    """
    LiDAR 포인트 클라우드를 이미지 평면에 투영 (배치 처리 지원)

    Args:
        pts_hom (torch.Tensor): 동차 좌표계 라이다 포인트 [B, 4] (x, y, z, 1)
        lidar2img (torch.Tensor): 변환 행렬 [B, 4, 4]
        eps (float): 0으로 나누기 방지용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 이미지 평면 UV 좌표와 깊이 [B, 3] (u, v, z)

    Raises:
        ValueError: 입력 차원이 유효하지 않은 경우
    """
    # 차원 검증
    if pts_hom.dim() != 2 or pts_hom.size(1) != 4:
        raise ValueError(f"pts_hom은 [B,4] 형태여야 합니다. 현재 형태: {pts_hom.shape}")
    if lidar2img.dim() != 3 or lidar2img.size(1) != 4 or lidar2img.size(2) != 4:
        raise ValueError(f"lidar2img은 [B,4,4] 형태여야 합니다. 현재 형태: {lidar2img.shape}")
    if pts_hom.size(0) != lidar2img.size(0):
        raise ValueError(f"배치 크기 불일치: pts_hom={pts_hom.size(0)}, lidar2img={lidar2img.size(0)}")

    # 차원 확장 및 변환
    lidar2img = lidar2img.float()
    pts_hom_ = pts_hom.unsqueeze(-1)  # [B,4,1]
    cam_points = torch.bmm(lidar2img, pts_hom_).squeeze(-1)  # [B,4]

    # 좌표 정규화
    u_coords = cam_points[:, 0] / (cam_points[:, 2] + eps)
    v_coords = cam_points[:, 1] / (cam_points[:, 2] + eps)
    z_depth = cam_points[:, 2]

    return torch.stack([u_coords, v_coords, z_depth], dim=1)  # [B,3]

def image_to_lidar_global(pred_uvz, gt_KT):
    """
    이미지 좌표(u, v, z)를 라이다 전역 좌표계로 변환
    Args:
        pred_uvz: [B, N, 3] (u, v, z) 이미지 좌표
        gt_KT: [B, 4, 4] 이미지->라이다 변환 행렬
    Returns:
        xyz_global: [B, N, 3] 라이다 전역 좌표
    """
    B, N, _ = pred_uvz.shape
    device = pred_uvz.device
    
    # 역변환 행렬 계산 [B,4,4]
    inverse_gt_kt = torch.inverse(gt_KT)
    
    # 정규화 좌표 계산 (u*z, v*z, z)
    uvz_scaled = pred_uvz.clone()
    uvz_scaled[..., :2] *= uvz_scaled[..., 2:3]  # [u*z, v*z, z]
    
    # 동차 좌표 변환 [B, N, 4]
    uvz_homogeneous = torch.cat([
        uvz_scaled, 
        torch.ones(B, N, 1, device=device)
    ], dim=-1)
    
    # 회전 변환: [B,3,3] @ [B,3,N] → [B,3,N]
    rotation = inverse_gt_kt[:, :3, :3]
    translation = inverse_gt_kt[:, :3, 3]
    
    # 행렬 곱 수행 (배치 처리)
    xyz_rot = torch.matmul(
        rotation, 
        uvz_homogeneous[..., :3].permute(0, 2, 1)
    )  # [B,3,N]
    
    # 병렬 이동 적용 및 차원 재정렬
    xyz_lidar = (xyz_rot.permute(0, 2, 1) + translation.unsqueeze(1))
    
    return xyz_lidar

def selected_image_to_lidar_global(pred_uvz, gt_KT):
    """
    이미지 좌표(u, v, z) + 카메라 인덱스 → 라이다 전역 좌표 변환
    Args:
        pred_uvz: [B, N, 4] (cam_idx, u, v, z)
        gt_KT: [Total_Cams, 4, 4] 전체 카메라 변환 행렬 풀
    Returns:
        xyz_global: [B, N, 3] 라이다 좌표
    """
    device = pred_uvz.device
    B, N, _ = pred_uvz.shape
    
    # 1. 데이터 구조 재구성
    flat_uvz = pred_uvz.view(-1, 4)          # [B*N, 4]
    camera_indices = flat_uvz[:, 0].long()   # [B*N]
    
    # 2. 각 포인트에 해당하는 KT 행렬 선택
    selected_KT = gt_KT[camera_indices]      # [B*N, 4, 4]
    
    # 3. 역변환 행렬 계산
    inverse_KT = torch.inverse(selected_KT)  # [B*N, 4, 4]
    
    # 4. 좌표 변환 수행
    uvz_scaled = flat_uvz[:, 1:].clone()     # [B*N, 3]
    uvz_scaled[:, :2] *= uvz_scaled[:, 2:]   # u*z, v*z
    
    # 5. 동차 좌표 변환
    uvz_homo = torch.cat([
        uvz_scaled,
        torch.ones((B*N, 1), device=device)
    ], dim=1)                                # [B*N, 4]
    
    # 6. 행렬 연산 (배치 처리)
    xyz_rot = torch.bmm(
        inverse_KT[:, :3, :3],               # [B*N, 3, 3]
        uvz_homo[:, :3].unsqueeze(-1)        # [B*N, 3, 1]
    ).squeeze(-1)                            # [B*N, 3]
    
    xyz_global = xyz_rot + inverse_KT[:, :3, 3]
    
    # 7. 원본 배치 형태 복원
    return xyz_global.view(B, N, 3)

def miscalib_transform(det_xyz, mis_T):
    # inverse_gt_kt = torch.inverse(gt_KT).float()
    
    list_xyz_global = []
    for cid in range(6):
        rotate_lidar2lidar = mis_T[cid]
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 1:]
            rotated_points = detection_xyz[:, :3].matmul(rotate_lidar2lidar[:3, :3].T) + rotate_lidar2lidar[:3, 3].unsqueeze(0)
            # points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
            list_xyz_global.append(rotated_points)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
    else:
        xyz_global_torch = torch.empty(0, 3, device=det_xyz.device)
    
    return xyz_global_torch

def miscalib_transform1(det_xyz, mis_extrinsic):
    list_xyz_global = []
    list_confidence = []
    list_cam_indices = []  # New list for camera indices
    
    for cid in range(6):
        RT = mis_extrinsic[cid]
        # 회전 행렬(R)과 이동 벡터(t) 분리
        # R = RT[:3, :3]  # 회전 행렬 전치 (LiDAR → 카메라 좌표계 변환)
        # t = RT[:3, 3]
        
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 1:4]  # Only take x,y,z coordinates
            confidence = det_xyz[mask, 4]  # Get confidence scores
            cam_indices = det_xyz[mask, 0]  # Get camera indices

            # 회전 적용: (N,3) @ (3,3) → (N,3)
            # rotated_points = detection_xyz @ R
            points_hom = torch.cat([detection_xyz, torch.ones_like(detection_xyz[:, :1])], dim=1)
            # Apply inverse transformation: LiDAR → Camera
            rotated_points = (RT @ points_hom.T).T
            rotated_points = rotated_points[:,:3]
            # 이동 적용: (N,3) + (1,3)
            # translated_points = rotated_points + t.unsqueeze(0)
            # # Z축 반전 (카메라 좌표계 방향 보정)
            # translated_points[:, 2] *= -1
            
            list_xyz_global.append(rotated_points)
            list_confidence.append(confidence)
            list_cam_indices.append(cam_indices)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)
        cam_indices_torch = torch.cat(list_cam_indices, dim=0).unsqueeze(1)
        
        # Concatenate camera indices, rotated points, and confidence scores
        xyz_global_torch = torch.cat([cam_indices_torch, xyz_global_torch, confidence_torch], dim=1)
    else:
        xyz_global_torch = torch.empty(0, 5, device=det_xyz.device)  # [0, 5] for cam_id,x,y,z,confidence
    
    return xyz_global_torch

def miscalib_transform2(det_xyz, mis_Rt):
    list_xyz_global = []
    list_confidence = []
    list_cam_indices = []  
    list_ob_indices = []
    
    for cid in range(6):
        Rt_perturb = mis_Rt[cid]
        mask = (det_xyz[:, 0] == cid)
        if mask.any():
            detection_xyz = det_xyz[mask, 2:5]  
            confidence = det_xyz[mask, 5]  
            cam_indices = det_xyz[mask, 0]
            ob_indices = det_xyz[mask, 1]

            # Convert to homogeneous coordinates and apply inverse
            points_hom = torch.cat([
                detection_xyz, 
                torch.ones_like(detection_xyz[:, :1])
            ], dim=1)
            
            # LiDAR → LiDAR
            perturbed_points = (Rt_perturb @ points_hom.T).T[:,:3]
            # rotated = detection_xyz @ RT_perturb[:3,:3].T
            # disturbed_xyz = rotated + RT_perturb[:3, 3]
            
            list_xyz_global.append(perturbed_points)
            list_confidence.append(confidence)
            list_cam_indices.append(cam_indices)
            list_ob_indices.append(ob_indices)
    
    if list_xyz_global:
        xyz_global_torch = torch.cat(list_xyz_global, dim=0)
        confidence_torch = torch.cat(list_confidence, dim=0).unsqueeze(1)
        cam_indices_torch = torch.cat(list_cam_indices, dim=0).unsqueeze(1)
        ob_indices_torch = torch.cat(list_ob_indices, dim=0).unsqueeze(1)
        
        xyz_global_torch = torch.cat(
            [cam_indices_torch,ob_indices_torch, xyz_global_torch, confidence_torch], 
            dim=1
        )
    else:
        xyz_global_torch = torch.empty(0, 6, device=det_xyz.device)
    
    return xyz_global_torch

def display_nonzero_depthmap(result_list, original_image):
    
    fig, axes = plt.subplots(1, len(result_list), figsize=(20, 5))
    if len(result_list) == 1:
        axes = [axes]
    
    for cam_idx, camera_results in enumerate(result_list):
        ax = axes[cam_idx]
        ax.imshow(original_image[cam_idx].permute(1,2,0).cpu().numpy())
        ax.set_title(f'Camera {cam_idx + 1}')
        
        for bbox_points in camera_results:
            if len(bbox_points) > 0:
                x = bbox_points[:, 0].cpu().numpy()
                y = bbox_points[:, 1].cpu().numpy()
                z = bbox_points[:, 2].cpu().numpy()
                
                scatter = ax.scatter(x, y, c=z, cmap='viridis', s=0.5, alpha=0.5)
        
        ax.set_xlim(0, original_image.shape[3])
        ax.set_ylim(original_image.shape[2], 0)
        ax.axis('off')
    
    plt.colorbar(scatter, ax=axes[-1], label='Depth')
    plt.tight_layout()
    plt.savefig('detection_bbox.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def normalize_point_cloud(point_cloud):
    # 포인트 클라우드의 형태: [batch_size, num_points, 3]
    
    # 1. 중심 이동: 각 포인트 클라우드의 평균을 계산하고 빼줍니다
    centroid = torch.mean(point_cloud, dim=1, keepdim=True)
    point_cloud = point_cloud - centroid

    # 2. 스케일 정규화: 원점으로부터의 최대 거리를 계산합니다
    max_distance = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=-1)), dim=1, keepdim=True)[0]
    
    # 3. 포인트 클라우드를 최대 거리로 나누어 [-1, 1] 범위로 정규화합니다
    normalized_point_cloud = point_cloud / max_distance.unsqueeze(-1)

    return normalized_point_cloud

def minmax_normalize_uvz(uvz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    UVZ 좌표를 채널별 민맥스 정규화 (0~1 범위)

    Args:
        uvz (torch.Tensor): 원본 UVZ 좌표 [B, 3]
        eps (float): 0 나누기 방지용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 정규화된 UVZ 좌표 [B, 3]
    """
    if uvz.dim() != 2 or uvz.size(1) != 3:
        raise ValueError(f"uvz는 [B,3] 형태여야 합니다. 현재 형태: {uvz.shape}")

    # 채널별 최소/최대 계산
    min_vals = uvz.min(dim=0, keepdim=True)[0]  # [1,3]
    max_vals = uvz.max(dim=0, keepdim=True)[0]  # [1,3]
    
    # 정규화
    uvz_norm = (uvz - min_vals) / (max_vals - min_vals + eps)
    return uvz_norm , min_vals , max_vals

def minmax_denormalize_uvz(uvz_norm: torch.Tensor, 
                          min_vals: torch.Tensor, 
                          max_vals: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    정규화된 UVZ 좌표를 원본 스케일로 역변환

    Args:
        uvz_norm (torch.Tensor): 정규화된 UVZ [B,3]
        min_vals (torch.Tensor): 원본 최솟값 [1,3] or [3]
        max_vals (torch.Tensor): 원본 최댓값 [1,3] or [3]
        eps (float): 수치 안정성용 작은 값 (기본값: 1e-6)

    Returns:
        torch.Tensor: 역정규화된 UVZ 좌표 [B,3]
    """
    # 차원 검증
    if uvz_norm.dim() != 2 or uvz_norm.size(1) != 3:
        raise ValueError(f"uvz_norm은 [B,3] 형태여야 합니다. 현재: {uvz_norm.shape}")
    if min_vals.shape != max_vals.shape:
        raise ValueError(f"min/max 차원 불일치: min={min_vals.shape}, max={max_vals.shape}")
    if min_vals.numel() != 3 or max_vals.numel() != 3:
        raise ValueError(f"min/max는 3개 채널 값을 가져야 합니다. 현재: min={min_vals.numel()}, max={max_vals.numel()}")

    # 차원 조정 (벡터 → 행렬)
    min_vals = min_vals.view(1, -1)  # [3] → [1,3]
    max_vals = max_vals.view(1, -1)  # [3] → [1,3]

    # 역정규화 계산
    range_vals = max_vals - min_vals + eps
    uvz_orig = uvz_norm * range_vals + min_vals
    
    return uvz_orig

def corrs_normalization(corrs,origin_img_shape=(900,1600,3),):
    
    # corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:,:, 0] = corrs[:,:, 0]/origin_img_shape[1] 
    # corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:,:, 1] = corrs[:,:, 1]/origin_img_shape[0] 
    if corrs[:,:, 2].numel() > 0:
        corrs[:,:, 2] = (corrs[:,:, 2]-torch.min(corrs[:,:, 2]))/(torch.max(corrs[:,:, 2]) - torch.min(corrs[:,:, 2]))
    else :
        corrs[:,:, 2] = (corrs[:,:, 2]-0)/(60 - 0)
    # corrs[:, 3] = (0.5*corrs[:, 3])/1280 + 0.5
    corrs[:,:, 3] = corrs[:,:, 3]/origin_img_shape[1]         
    # corrs[:, 4] = (0.5*corrs[:, 4])/384
    corrs[:,:, 4] = corrs[:,:, 4]/origin_img_shape[0]
    if corrs[:,:, 5].numel() > 0:
        corrs[:,:, 5] = (corrs[:,:, 5]-torch.min(corrs[:,:, 5]))/(torch.max(corrs[:,:, 5]) - torch.min(corrs[:,:, 5])) 
    else :
        corrs[:,:, 5] = (corrs[:,:, 5]-0)/(60 - 0)
    
    return corrs

def corrs_denormalization(corrs_modi, origin_img_shape=(900, 1600, 3)):
    # x 좌표 역정규화
    corrs = corrs_modi.clone().detach()
    corrs[:, :, 0] = corrs[:, :, 0] * origin_img_shape[1]
    
    # y 좌표 역정규화
    corrs[:, :, 1] = corrs[:, :, 1] * origin_img_shape[0]
    
    # z 좌표 역정규화
    if corrs[:, :, 2].numel() > 0:
        min_z = torch.min(corrs[:, :, 2])
        max_z = torch.max(corrs[:, :, 2])
        corrs[:, :, 2] = corrs[:, :, 2] * (max_z - min_z) + min_z
    else:
        corrs[:, :, 2] = corrs[:, :, 2] * 60
       
    return corrs

def points2depthmap(points, height, width ,downsample=1): 
    device = points.device
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5], # original
        # 'depth': [1.0, 80.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=points.dtype, device=device)
    coor = torch.round(points[:, :2] / downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth

    # 최종 유효 인덱스 계산
    valid_indices = torch.where(kept1)[0][kept2]
    
    return depth_map, coor, depth , valid_indices

def dense_map_gpu_optimized(Pts, n, m, grid):
    device = Pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8  # 작은 값 추가하여 0으로 나누는 상황 방지
    # import pdb; pdb.set_trace()
    # 초기 텐서를 GPU로 이동
    mX = torch.full((m, n), float('inf'), dtype=Pts.dtype, device=device)
    mY = torch.full((m, n), float('inf'), dtype=Pts.dtype, device=device)
    mD = torch.zeros((m, n), dtype=Pts.dtype, device=device)

    mX_idx = Pts[1].clone().detach().to(dtype=torch.int32, device=device)
    mY_idx = Pts[0].clone().detach().to(dtype=torch.int32, device=device)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=Pts.dtype, device=device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=Pts.dtype, device=device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=Pts.dtype, device=device)

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
    out = torch.zeros((m, n), dtype=Pts.dtype, device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    # return out.cpu()  # 최종 결과를 CPU로 이동
    return out # 최종 결과를 GPU

def dense_map_from_depth_batch(lidar_depth_mis, grid=5, iterations=3):
    batch_size, H, W = lidar_depth_mis.shape
    device = lidar_depth_mis.device
    output = lidar_depth_mis.clone()
    
    # 1. 마스크 생성 (값이 0이 아닌 유효 포인트)
    valid_mask = (output > 0)
    
    # 2. 각 배치별 처리
    for b in range(batch_size):
        depth = output[b]
        mask = valid_mask[b]
        
        # 3. 여러 번 반복적용으로 점진적 dense화
        for _ in range(iterations):
            # 커널 크기 (ng x ng)
            ng = 2 * grid + 1
            
            # 이웃 픽셀 영향 계산을 위한 패딩
            padded_depth = F.pad(depth.unsqueeze(0).unsqueeze(0), 
                               (grid, grid, grid, grid), mode='constant', value=0)
            padded_mask = F.pad(mask.unsqueeze(0).unsqueeze(0).float(), 
                              (grid, grid, grid, grid), mode='constant', value=0)
            
            # 거리 기반 가중치 커널 생성
            y, x = torch.meshgrid(
                torch.arange(-grid, grid+1, device=device),
                torch.arange(-grid, grid+1, device=device)
            )
            dist = torch.sqrt(x**2 + y**2 + 1e-8)
            weight_kernel = (1.0 / dist**2).view(1, 1, ng, ng)
            
            # 마스크 가중치 적용 (유효 픽셀만 기여)
            # 합성곱으로 깊이값 가중 합 계산
            weighted_depth = F.conv2d(padded_depth * padded_mask, weight_kernel, padding=0)
            # 가중치 합 계산
            weight_sum = F.conv2d(padded_mask, weight_kernel, padding=0)
            
            # 새 마스크 (기존 유효점 + 이웃에 유효점이 있는 픽셀)
            new_mask = (weight_sum > 0).squeeze()
            
            # 가중 평균 계산 및 업데이트
            new_depth = depth.clone()
            # new_depth[new_mask] = (weighted_depth / (weight_sum + 1e-8))[0, 0][new_mask]
            new_depth[new_mask] = (weighted_depth / (weight_sum + 1e-8))[0, 0].to(new_depth.dtype)[new_mask]
            
            # 업데이트
            depth = new_depth
            mask = new_mask
        
        # 결과 저장
        output[b] = depth
    
    return output

def dense_map_from_depth_batch_v2(lidar_depth_mis, grid=5, iterations=3):
    # 1. shape 언패킹 및 디바이스 설정
    N, V, H, W = lidar_depth_mis.shape
    device = lidar_depth_mis.device
    
    # 2. 텐서 재구성: (N, V, H, W) -> (N*V, H, W)
    # 모든 깊이 맵을 하나의 큰 배치로 취급하여 병렬 처리
    depths = lidar_depth_mis.reshape(N * V, H, W)
    
    # 3. 마스크 생성
    mask = (depths > 0)
    
    # 4. 반복적 dense화 (모든 맵에 대해 동시에 수행)
    for _ in range(iterations):
        ng = 2 * grid + 1
        
        # conv2d를 위해 채널 차원 추가: (N*V, H, W) -> (N*V, 1, H, W)
        depths_ch = depths.unsqueeze(1)
        mask_ch = mask.unsqueeze(1).float()
        
        # 배치 전체에 대해 패딩 동시 적용
        padded_depths = F.pad(depths_ch, (grid, grid, grid, grid), mode='constant', value=0)
        padded_masks = F.pad(mask_ch, (grid, grid, grid, grid), mode='constant', value=0)
        
        # 가중치 커널 생성 (모든 맵에 동일하게 사용)
        y, x = torch.meshgrid(
            torch.arange(-grid, grid + 1, device=device, dtype=torch.float32),
            torch.arange(-grid, grid + 1, device=device, dtype=torch.float32),
            indexing='ij'
        )
        dist = torch.sqrt(x**2 + y**2 + 1e-8)
        weight_kernel = (1.0 / dist**2).view(1, 1, ng, ng)
        
        # 배치 전체에 대해 합성곱 동시 적용
        weighted_depths = F.conv2d(padded_depths * padded_masks, weight_kernel, padding=0)
        weight_sums = F.conv2d(padded_masks, weight_kernel, padding=0)
        
        # 업데이트할 영역 마스크 생성
        updatable_mask = (weight_sums > 1e-8).squeeze(1) # (N*V, H, W)
        
        # 0으로 나누는 것을 방지하며 가중 평균 계산 및 업데이트
        avg_depths = (weighted_depths.squeeze(1) / weight_sums.squeeze(1))
        depths[updatable_mask] = avg_depths.to(depths.dtype)[updatable_mask]
        
        # 다음 반복을 위해 메인 마스크 업데이트
        mask = (depths > 0)

    # 5. 원래 shape (N, V, H, W)로 복원 후 반환
    return depths.reshape(N, V, H, W)

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
    colormapped_tensor = torch.from_numpy(colormapped_im)
    return colormapped_tensor

def batch_colormap(disp_batch, cmap='magma'):
    """
    GPU를 사용하여 4D 텐서([N, V, H, W])에 컬러맵을 적용하는 함수.
    각 뷰(V)별로 독립적인 정규화를 수행합니다.
    """
    # 1. 입력 텐서의 shape 언패킹 및 LUT 로드
    n, v, h, w = disp_batch.shape
    device = disp_batch.device
    
    # 컬러맵 LUT(Look-Up Table)를 GPU에 미리 로드
    cmap_func = plt.get_cmap(cmap, 256)
    cmap_lut = torch.tensor(cmap_func.colors[:, :3], dtype=torch.float32, device=device)  # [256, 3]

    # 2. 텐서 재구성 및 정규화
    # (N, V, H, W) -> (N*V, H, W) 형태로 변경하여 모든 뷰를 하나의 배치로 처리
    disp_reshaped = disp_batch.reshape(n * v, h, w)
    
    eps = 1e-8

    # 각 뷰(이미지)별로 최소/최대값을 독립적으로 계산
    disp_flat = disp_reshaped.view(n * v, -1)
    vmin = disp_flat.min(dim=1, keepdim=True)[0]  # [N*V, 1]
    vmax = disp_flat.max(dim=1, keepdim=True)[0]  # [N*V, 1]
    
    # 브로드캐스팅을 위해 vmin, vmax를 (N*V, 1, 1) 형태로 변경
    vmin = vmin.view(n * v, 1, 1)
    vmax = vmax.view(n * v, 1, 1)

    # 각 뷰를 0-1 사이로 정규화
    normalized = (disp_reshaped - vmin) / (vmax - vmin + eps)
    
    # 3. LUT 인덱싱
    # 0-255 범위의 정수 인덱스로 변환
    indices = (normalized * 255).clamp(0, 255).long()  # shape: [N*V, H, W]
    
    # LUT를 사용하여 컬러값 조회
    colored_reshaped = cmap_lut[indices]  # shape: [N*V, H, W, 3]
    
    # 4. 최종 차원 재배열
    # (N*V, H, W, 3) -> (N*V, 3, H, W)
    colored_permuted = colored_reshaped.permute(0, 3, 1, 2)
    
    # 최종적으로 (N, V, 3, H, W) 형태로 복원하여 반환
    return colored_permuted.reshape(n, v, 3, h, w)

def differentiable_colormap(disp_batch, cmap='magma'):
    """미분 가능한 컬러맵 변환 함수"""
    # 1. LUT 초기화
    magma_cmap = plt.get_cmap(cmap, 256)
    magma_lut = torch.tensor(magma_cmap.colors[:, :3], 
                           dtype=torch.float32, 
                           device=disp_batch.device)  # [256, 3]

    # 2. 입력 데이터 형상 추출
    b, h, w = disp_batch.shape
    eps = 1e-8  # 분모 0 방지

    # 3. 최소/최대 계산 (배치별 독립적)
    disp_flat = disp_batch.view(b, -1)
    vmin = disp_flat.min(dim=1).values.view(b, 1, 1)
    vmax = disp_flat.max(dim=1).values.view(b, 1, 1)

    # 4. 정규화 (0~1 범위)
    normalized = (disp_batch - vmin) / (vmax - vmin + eps)  # [B, H, W]

    # 5. 연속 인덱스 계산 (0~254)
    indices = normalized * 254  # 255개 색상 → 254 구간

    # 6. 선형 보간
    lower = torch.floor(indices).long()  # [B, H, W]
    upper = torch.ceil(indices).long()   # [B, H, W]
    alpha = indices - lower              # [B, H, W]

    # 7. 색상 보간 (벡터화 연산)
    lower_color = magma_lut[lower]       # [B, H, W, 3]
    upper_color = magma_lut[upper]       # [B, H, W, 3]
    colored = (1 - alpha.unsqueeze(-1)) * lower_color + alpha.unsqueeze(-1) * upper_color

    # 8. 차원 재배열 (BCHW)
    return colored.permute(0, 3, 1, 2)  # [B, 3, H, W]


def denormalize_points(normal_points, z_min=None, z_max=None):
    """
    정규화된 UVZ 좌표를 원본 좌표계로 변환
    Args:
        normal_points: [N,3] 정규화된 텐서 (u, v, z)
        z_min: 원본 z 최소값 (없을 경우 기본값 0 사용)
        z_max: 원본 z 최대값 (없을 경우 기본값 80 사용)
    Returns:
        denorm_points: [N,3] 역정규화된 텐서
    """
    denorm_points = normal_points.clone()
    
    # U 좌표 복원: [0.5~1.5) → [0~640)
    denorm_points[... , 0] = (denorm_points[..., 0] - 0.5) * 1280
    
    # V 좌표 복원: [0~1) → [0~192)
    denorm_points[..., 1] = denorm_points[..., 1] * 192
    
    # Z 좌표 복원
    if z_min is not None and z_max is not None:
        z_range = z_max - z_min
        denorm_points[..., 2] = denorm_points[..., 2] * z_range + z_min
    else:  # 기본 범위 사용 (0~80)
        denorm_points[..., 2] = denorm_points[..., 2] * 80
        
    return denorm_points

def pixel_to_normalized(uv_pixel, intrinsics):
    """ 배치 처리된 픽셀 → 정규화 좌표 변환 """
    # uv_pixel: [B, N, 5] (batch, num_points, [cam, obj, u, v, z])
    # intrinsics: [O,4,4] (객체별 내부 파라미터)
    
    # 1. 배치 차원 병합
    B, N, _ = uv_pixel.shape
    uv_flat = uv_pixel.view(-1, 5)  # [B*N, 5]
    
    # 2. 객체 ID 기반 파라미터 선택
    obj_ids = uv_flat[:, 1].long()  # [B*N]
    obj_intrinsics = intrinsics[obj_ids]  # [B*N,4,4]
    
    # 3. 파라미터 추출
    fx = obj_intrinsics[:, 0, 0]  # [B*N]
    fy = obj_intrinsics[:, 1, 1]  # [B*N]
    cx = obj_intrinsics[:, 0, 2]  # [B*N]
    cy = obj_intrinsics[:, 1, 2]  # [B*N]
    
    # 4. 좌표 분해 및 계산
    u = uv_flat[:, 2]  # [B*N]
    v = uv_flat[:, 3]  # [B*N]
    z = uv_flat[:, 4]  # [B*N]
    
    u_norm = (u - cx) / fx
    v_norm = (v - cy) / fy
    
    # 5. 배치 차원 복원
    normalized = torch.stack([
        uv_flat[:, 0],  # cam
        uv_flat[:, 1],  # obj
        u_norm,
        v_norm,
        z
    ], dim=1).view(B, N, 5)  # [B, N, 5]
    
    return normalized

def center2lidar_batch(center_pred, intrinsics, extrinsics):
    """
    center_pred: [B, N, 5] (cam_index, obj_id, u, v, z)
    intrinsics: [O,4,4] (객체별 내부 파라미터)
    extrinsics: [O,4,4] (객체별 외부 파라미터)
    """
    B, N, _ = center_pred.shape
    device = center_pred.device
    
    # 1. 객체 ID 추출 및 파라미터 선택
    obj_ids = center_pred[:, :, 1].long()  # [B, N]
    obj_ids_flat = obj_ids.view(-1)  # [B*N]
    
    # 객체별 파라미터 선택 및 차원 조정
    selected_intrinsics = intrinsics[obj_ids_flat].view(B, N, 4, 4)  # [B, N, 4, 4]
    selected_extrinsics = extrinsics[obj_ids_flat].view(B, N, 4, 4)  # [B, N, 4, 4]

    # 2. 좌표 변환 준비
    u = center_pred[:, :, 2]  # [B, N]
    v = center_pred[:, :, 3]  # [B, N]
    z = center_pred[:, :, 4]  # [B, N]
    
    # 3. 이미지 좌표계 → 라이다 좌표계 변환
    center_img = torch.stack([
        u * z,
        v * z,
        z,
        torch.ones_like(z)
    ], dim=-1)  # [B, N, 4]

    # 4. 변환 행렬 계산
    lidar2img = torch.matmul(
        selected_intrinsics,
        selected_extrinsics.transpose(2, 3)  # [B, N, 4, 4]
    )
    
    # 5. 역변환 행렬 계산
    img2lidar = torch.inverse(lidar2img)  # [B, N, 4, 4]

    # 6. 좌표 변환 수행
    center_lidar = torch.matmul(
        img2lidar,
        center_img.unsqueeze(-1)  # [B, N, 4, 1]
    ).squeeze(-1)[:, :, :3]  # [B, N, 3]

    center_lidar_with_index = torch.cat([center_pred[...,0:2],center_lidar],dim=2)

    return center_lidar_with_index,center_lidar, lidar2img

def differentiable_center2lidar(center_pred, intrinsics, extrinsics, eps=1e-6):
    """
    개선사항:
    1. 미분 가능한 객체 파라미터 선택
    2. 수치 안정성 강화된 역행렬 계산
    3. 배치 차원 보존 연산
    """
    B, N, _ = center_pred.shape
    device = center_pred.device
    
    # 1. 객체 ID 추출 (정수 변환 시 그래디언트 차단)
    with torch.no_grad():  # 객체 ID는 미분 흐름에서 제외
        obj_ids = center_pred[:, :, 1].long()  # [B, N]
    
    # 2. 미분 가능한 파라미터 선택 (One-hot 기반)
    obj_ids_flat = obj_ids.view(-1)  # [B*N]
    one_hot = torch.nn.functional.one_hot(obj_ids_flat, intrinsics.size(0))  # [B*N, O]
    
    # 3. 행렬 재구성 (배치 차원 보존)
    selected_intrinsics = torch.matmul(
        one_hot.unsqueeze(1).double(),  # [B*N, 1, O]
        intrinsics.view(intrinsics.size(0), -1)  # [O, 16]
    ).view(B, N, 4, 4)  # [B, N, 4, 4]
    
    selected_extrinsics = torch.matmul(
        one_hot.unsqueeze(1).double(),
        extrinsics.view(extrinsics.size(0), -1)
    ).view(B, N, 4, 4)

    # 4. 좌표 변환 준비 (기존 코드 유지)
    u = center_pred[:, :, 2]
    v = center_pred[:, :, 3]
    z = center_pred[:, :, 4]
    
    center_img = torch.stack([u*z, v*z, z, torch.ones_like(z)], dim=-1)
    
    # 5. 안정화된 역행렬 계산
    lidar2img = torch.matmul(selected_intrinsics, selected_extrinsics.transpose(2,3))
    lidar2img_reg = lidar2img + eps * torch.eye(4, device=device)  # 정칙화
    
    # 6. Pseudo-inverse 대체 (미분 가능)
    img2lidar = torch.linalg.pinv(lidar2img_reg)
    
    # 7. 변환 수행 (기존 코드 유지)
    center_lidar = torch.matmul(
        img2lidar,
        center_img.unsqueeze(-1)
    ).squeeze(-1)[:, :, :3]
    
    center_lidar_with_index = torch.cat([center_pred[...,0:2], center_lidar], dim=2)
    
    return center_lidar_with_index, center_lidar, lidar2img


# def draw_correspondences(trimed_corrs, sbs_img, save_path='correspond.jpg'):
#     """정규화 좌표 기반 시각화 (0~1 범위 입력 필요)"""
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # 1. 이미지 전처리
#     img_tensor = sbs_img# [3, 192, 1280]
#     denorm_img = img_tensor / 2 + 0.5  # 정규화 해제
#     img_np = denorm_img.permute(1, 2, 0).cpu().numpy()
    
#     # 2. 좌표 추출 및 스케일 복원
#     H, W = img_np.shape[:2]
#     left_pts = trimed_corrs[:, :2].detach().cpu().numpy()  # 정규화 좌표 [N,2] (0~1)
#     right_pts = trimed_corrs[:, 2:].detach().cpu().numpy()
    
#     # 3. 정규화 → 픽셀 좌표 변환 (원본 알고리즘 반영)
#     left_pts[:, 0] = left_pts[:, 0] * 640  # u = (norm_u - 0.5)*640 
#     left_pts[:, 1] = left_pts[:, 1] * 192  # v = norm_v * 192
#     right_pts[:, 0] = (right_pts[:, 0] - 0.5) * 640 + 640  # 우측 오프셋 적용
#     right_pts[:, 1] = right_pts[:, 1] * 192

#     # 4. 좌표 클리핑 및 필터링
#     left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W-1)
#     left_pts[:, 1] = np.clip(left_pts[:, 1], 0, H-1)
#     right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W-1)
#     right_pts[:, 1] = np.clip(right_pts[:, 1], 0, H-1)
    
#     valid_mask = ~(np.isnan(left_pts).any(axis=1) | np.isnan(right_pts).any(axis=1))
#     left_pts = left_pts[valid_mask]
#     right_pts = right_pts[valid_mask]

#     # 5. 시각화
#     plt.figure(figsize=(20, 6))
#     plt.imshow(img_np)
    
#     # 좌측 포인트 (청색)
#     plt.scatter(left_pts[:,0], left_pts[:,1], 
#                 c='cyan', s=30, edgecolors='k', linewidth=0.8, label='Left Points')
#     # 우측 포인트 (자홍색)
#     plt.scatter(right_pts[:,0], right_pts[:,1], 
#                 c='magenta', s=30, edgecolors='k', linewidth=0.8, label='Right Points')
    
#     # 연결선 그리기 (옵션)
#     for l, r in zip(left_pts, right_pts):
#         plt.plot([l[0], r[0]], [l[1], r[1]], 
#                 color='yellow', linestyle='--', linewidth=1.5, alpha=0.4)

#     plt.axis('off')
#     plt.legend(loc='upper right', prop={'size': 12})
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Correspondence image saved to {save_path}")

def draw_correspondences(
    trimed_corrs, 
    sbs_img, 
    save_path='correspondence.jpg', 
    bboxes_to_draw=None, 
    score_thr=0.3
):
    """
    Side-by-Side 이미지에 대응점과 Bounding Box를 함께 시각화하여 저장합니다.
    BBox 좌표를 [900, 1600] -> [192, 640] 스케일로 정확하게 변환합니다.
    """
    # 1. 이미지 텐서 전처리
    denorm_img = sbs_img / 2 + 0.5
    img_np = denorm_img.permute(1, 2, 0).cpu().numpy()
    H_img, W_img, _ = img_np.shape # 현재 시각화될 SBS 이미지의 높이(192), 너비(1280)

    # SBS 이미지의 왼쪽 절반에 해당하는 너비 (640)
    LEFT_IMG_WIDTH = W_img / 2 
    # SBS 이미지의 높이 (192)
    LEFT_IMG_HEIGHT = H_img

    # 원본 BBox의 기준 해상도
    ORIGINAL_BBOX_WIDTH = 1600
    ORIGINAL_BBOX_HEIGHT = 900

    # BBox 스케일링 비율 계산
    scale_x = LEFT_IMG_WIDTH / ORIGINAL_BBOX_WIDTH # 640 / 1600 = 0.4
    scale_y = LEFT_IMG_HEIGHT / ORIGINAL_BBOX_HEIGHT # 192 / 900 = 0.2133...

    # 2. Matplotlib 시각화 준비
    fig, ax = plt.subplots(1, figsize=(20, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img_np)

    # 3. 대응점(Correspondence Points) 그리기 (기존 로직 유지)
    if trimed_corrs is not None and trimed_corrs.shape[0] > 0:
        left_pts = trimed_corrs[:, :2].detach().cpu().numpy()
        right_pts = trimed_corrs[:, 2:].detach().cpu().numpy()
        
        # 정규화 좌표를 SBS 이미지의 픽셀 좌표로 변환
        left_pts[:, 0] = left_pts[:, 0] * LEFT_IMG_WIDTH
        left_pts[:, 1] = left_pts[:, 1] * LEFT_IMG_HEIGHT
        right_pts[:, 0] = (right_pts[:, 0] - 0.5) * LEFT_IMG_WIDTH + LEFT_IMG_WIDTH
        right_pts[:, 1] = right_pts[:, 1] * LEFT_IMG_HEIGHT

        valid_mask = ~(np.isnan(left_pts).any(axis=1) | np.isnan(right_pts).any(axis=1))
        left_pts, right_pts = left_pts[valid_mask], right_pts[valid_mask]

        ax.scatter(left_pts[:, 0], left_pts[:, 1], c='cyan', s=30, edgecolors='k', linewidth=0.8, label='Left Points')
        ax.scatter(right_pts[:, 0], right_pts[:, 1], c='magenta', s=30, edgecolors='k', linewidth=0.8, label='Right Points')
        for l, r in zip(left_pts, right_pts):
            ax.plot([l[0], r[0]], [l[1], r[1]], color='yellow', linestyle='--', linewidth=1.5, alpha=0.4)

    # # 4. Bounding Box 그리기 (수정된 로직)
    # if bboxes_to_draw is not None and bboxes_to_draw.shape[0] > 0:
    #     bboxes = bboxes_to_draw.clone()

    #     # ====================================================================
    #     # ===== 여기가 BBox 좌표 스케일링 및 이동 로직입니다. =====
    #     # ====================================================================
    #     # X 좌표 스케일링
    #     bboxes[:, [0, 2]] *= scale_x
    #     # Y 좌표 스케일링
    #     bboxes[:, [1, 3]] *= scale_y
        
    #     for bbox_data in bboxes:
    #         coords, score, label = bbox_data[:4], bbox_data[4], int(bbox_data[5].item())
    #         if score < score_thr:
    #             continue
            
    #         x1, y1, x2, y2 = coords.cpu().numpy()
    #         width, height = x2 - x1, y2 - y1
            
    #         rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='blue', facecolor='none')
    #         ax.add_patch(rect)
            
    #         # 텍스트 위치도 스케일링된 좌표에 맞춰 조정
    #         ax.text(x1, y1 - 5, f'L:{label} | {score:.2f}', 
    #                 color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # 5. 최종 이미지 저장
    ax.set_aspect('equal') # 비율 유지
    ax.axis('off')
    # ax.legend(loc='upper right', prop={'size': 12})
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Correspondence and BBox image saved to {save_path}")

def geometric_propagation(depth_map, iterations=3):
    """
    깊이 맵의 기하학적 일관성을 보존하는 전파 알고리즘
    """
    for _ in range(iterations):
        # x축 방향 전파
        depth_diff = torch.abs(depth_map[:,:,1:] - depth_map[:,:,:-1])
        valid_mask = (depth_diff < 1.0).float()  # 1m 이내 차이만 허용
        depth_map[:,:,1:] = depth_map[:,:,:-1]*valid_mask + depth_map[:,:,1:]*(1-valid_mask)
        
        # y축 방향 전파
        depth_diff = torch.abs(depth_map[:,1:,:] - depth_map[:,:-1,:])
        valid_mask = (depth_diff < 1.0).float()
        depth_map[:,1:,:] = depth_map[:,:-1,:]*valid_mask + depth_map[:,1:,:]*(1-valid_mask)
        
    return depth_map


def trim_detection_points(detection_xyz_lidar, trim_count=200):
    """
    cam_id 분포를 유지하며 포인트 트리밍
    Args:
        detection_xyz_lidar: [N,6] Tensor (cam_id, obj_id, x, y, z, confidence_score)
        trim_count: 목표 포인트 개수
    Returns:
        trimmed_points: [trim_count,6] Tensor
    """
    if len(detection_xyz_lidar) <= trim_count:
        return detection_xyz_lidar
    
    # cam_id 분포 계산
    unique_cams, counts = torch.unique(detection_xyz_lidar[:,0], return_counts=True)
    cam_distribution = counts.float() / counts.sum()
    
    # 각 cam_id별 할당 개수 계산
    allocations = (cam_distribution * trim_count).long()
    remainder = trim_count - allocations.sum()
    
    # 나머지 분배 (큰 소수점 순으로)
    fracs = (cam_distribution * trim_count) - allocations.float()
    _, indices = torch.sort(fracs, descending=True)
    for i in range(remainder):
        allocations[indices[i]] += 1

    # 포인트 선택
    selected = []
    for cam, alloc in zip(unique_cams, allocations):
        mask = detection_xyz_lidar[:,0] == cam
        candidates = detection_xyz_lidar[mask]
        if len(candidates) > alloc:
            idx = torch.randperm(len(candidates))[:alloc]
            selected.append(candidates[idx])
        else:
            selected.append(candidates)
    
    return torch.cat(selected)

# def generate_random_points(detection_xyz_lidar, target_count=200):
#     num_existing = len(detection_xyz_lidar)
#     if num_existing >= target_count:
#         return detection_xyz_lidar
    
#     device = detection_xyz_lidar.device  # 디바이스 정보 추출
#     dtype = detection_xyz_lidar.dtype    # 데이터 타입 추출

#     # 기존 데이터 통계 계산
#     x_min, x_max = detection_xyz_lidar[:,2].min(), detection_xyz_lidar[:,2].max()
#     y_min, y_max = detection_xyz_lidar[:,3].min(), detection_xyz_lidar[:,3].max()
#     z_min, z_max = detection_xyz_lidar[:,4].min(), detection_xyz_lidar[:,4].max()
    
#     # cam_id 분포 계산
#     unique_cams, counts = torch.unique(detection_xyz_lidar[:,0], return_counts=True)
#     cam_distribution = counts.float() / counts.sum()
    
#     # 생성할 포인트 수 계산
#     num_to_generate = target_count - num_existing
#     allocations = (cam_distribution * num_to_generate).long()
#     remainder = num_to_generate - allocations.sum()
    
#     # 나머지 분배
#     if remainder > 0:
#         allocations[0] += remainder
    
#     # 포인트 생성
#     new_points = []
#     for cam, alloc in zip(unique_cams, allocations):
#         if alloc == 0:
#             continue
            
#         # 좌표 생성 (디바이스/데이터타입 명시)
#         x = torch.empty(alloc, device=device, dtype=dtype).uniform_(x_min, x_max)
#         y = torch.empty(alloc, device=device, dtype=dtype).uniform_(y_min, y_max)
#         z = torch.empty(alloc, device=device, dtype=dtype).uniform_(z_min, z_max)
        
#         # 신규 obj_id 생성 (기존 최대값 +1부터 시작)
#         max_obj_id = detection_xyz_lidar[:,1].max()
#         obj_ids = torch.arange(
#             max_obj_id+1, 
#             max_obj_id+1+alloc, 
#             device=device, 
#             dtype=detection_xyz_lidar[:,1].dtype
#         )
        
#         # 텐서 조립
#         new = torch.stack([
#             torch.full((alloc,), cam, device=device, dtype=dtype),
#             obj_ids,
#             x,
#             y,
#             z,
#             torch.zeros(alloc, device=device, dtype=dtype)  # confidence_score=0
#         ], dim=1)
        
#         new_points.append(new)
    
#     return torch.cat([detection_xyz_lidar] + new_points, dim=0)

def generate_random_points(detection_xyz_lidar, target_count=200):
    num_existing = len(detection_xyz_lidar)
    if num_existing >= target_count:
        return detection_xyz_lidar
    
    device = detection_xyz_lidar.device  # 디바이스 정보 추출
    dtype = detection_xyz_lidar.dtype    # 데이터 타입 추출

    # 기존 데이터 통계 계산
    x_min, x_max = detection_xyz_lidar[:,2].min(), detection_xyz_lidar[:,2].max()
    y_min, y_max = detection_xyz_lidar[:,3].min(), detection_xyz_lidar[:,3].max()
    z_min, z_max = detection_xyz_lidar[:,4].min(), detection_xyz_lidar[:,4].max()
    
    # cam_id 분포 계산
    unique_cams, counts = torch.unique(detection_xyz_lidar[:,0], return_counts=True)
    cam_distribution = counts.float() / counts.sum()
    
    # 생성할 포인트 수 계산
    num_to_generate = target_count - num_existing
    allocations = (cam_distribution * num_to_generate).long()
    remainder = num_to_generate - allocations.sum()
    
    # 나머지 분배
    if remainder > 0:
        allocations[0] += remainder
    
    # 포인트 생성
    new_points = []
    
    # 생성 전략 수정: 단순 랜덤이 아닌 실제 분포 고려
    for cam, alloc in zip(unique_cams, allocations):
        if alloc == 0:
            continue
        
        # 실제 포인트 분포를 기반으로 샘플링
        existing_points = detection_xyz_lidar[detection_xyz_lidar[:, 0] == cam]
        
        if len(existing_points) >= 3:  # 충분한 포인트가 있는 경우
            # KDE 또는 GMM으로 분포 추정 후 샘플링
            mean = existing_points[:, 2:5].mean(dim=0)
            std = existing_points[:, 2:5].std(dim=0)
            
            # 가우시안 분포로 샘플링 (균등 분포 대신)
            xyz = torch.randn(alloc, 3, device=device, dtype=dtype)
            xyz = xyz * std + mean
            
            # 좌표값 추출
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        else:
            # 포인트가 적으면 기존 방식 사용
            x = torch.empty(alloc, device=device, dtype=dtype).uniform_(x_min, x_max)
            y = torch.empty(alloc, device=device, dtype=dtype).uniform_(y_min, y_max)
            z = torch.empty(alloc, device=device, dtype=dtype).uniform_(z_min, z_max)
            
        # 신규 obj_id 생성 (기존 최대값 +1부터 시작)
        max_obj_id = detection_xyz_lidar[:,1].max()
        obj_ids = torch.arange(
            max_obj_id+1, 
            max_obj_id+1+alloc, 
            device=device, 
            dtype=detection_xyz_lidar[:,1].dtype
        )
        
        # 텐서 조립
        new = torch.stack([
            torch.full((alloc,), cam, device=device, dtype=dtype),
            obj_ids,
            x, y, z,
            torch.zeros(alloc, device=device, dtype=dtype)  # confidence_score=0
        ], dim=1)
        
        new_points.append(new)
    
    return torch.cat([detection_xyz_lidar] + new_points, dim=0)


def trim_or_generate_points(detection_xyz_lidar, target_count=200):
    if len(detection_xyz_lidar) > target_count:
        return trim_detection_points(detection_xyz_lidar, target_count)
    elif len(detection_xyz_lidar) < target_count:
        return generate_random_points(detection_xyz_lidar, target_count)
    return detection_xyz_lidar

import torch

# def deduplicate_obj_ids(input_tensor):
#     # 입력 텐서 평탄화: [C, P, 5] -> [C*P, 5]
#     flat_data = input_tensor.view(-1, 5)
    
#     # obj_id 추출 및 정수형 변환 (인덱스 1)
#     obj_ids = flat_data[:, 1].to(torch.int64)
    
#     # 고유값과 역인덱스 추출
#     unique_obj_ids, inverse_indices = torch.unique(
#         obj_ids,
#         return_inverse=True,
#         sorted=True
#     )
    
#     # 첫 번째 발생 인덱스 계산
#     inv_sorted = inverse_indices.argsort()
#     counts = torch.bincount(inverse_indices)
#     tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
#     unique_indices = inv_sorted[tot_counts]
    
#     return flat_data[unique_indices]

def deduplicate_obj_ids(input_tensor):
    flat_data = input_tensor.view(-1, 5)
    obj_ids = flat_data[:, 1].to(torch.int64)
    
    # [수정 1] unique 값만 추출
    unique_obj_ids = torch.unique(obj_ids, sorted=True)
    
    last_occurrence = torch.zeros(len(unique_obj_ids), dtype=torch.long)
    
    # [수정 2] 각 고유 ID별 마지막 발생 위치 탐색
    for i, obj_id in enumerate(unique_obj_ids):
        matches = torch.where(obj_ids == obj_id)[0]
        if len(matches) > 0:
            last_occurrence[i] = matches[-1]
        else:
            last_occurrence[i] = -1  # 유효하지 않은 인덱스 처리
    
    # [수정 3] 유효한 인덱스만 필터링
    valid_mask = last_occurrence != -1
    return flat_data[last_occurrence[valid_mask]]

def differentiable_deduplicate(input_tensor, temperature=0.1):
    """Differentiable last-occurrence selection for object IDs"""
    flat_data = input_tensor.view(-1, 5)
    obj_ids = flat_data[:, 1].long()
    
    # 1. Create group masks using broadcasting
    unique_ids = torch.unique(obj_ids)
    mask = (obj_ids.unsqueeze(1) == unique_ids.unsqueeze(0))  # [N, U]
    
    # 2. Positional weighting using temporal scores
    indices = torch.arange(flat_data.size(0), device=obj_ids.device).float()
    logits = mask * indices.unsqueeze(1)  # [N, U]
    
    # 3. Temperature-controlled softmax for sharp selection
    weights = torch.softmax(logits / temperature, dim=0) * mask.float()
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)  # Normalize
    
    # 4. Differentiable weighted sum (approximates last-occurrence selection)
    return torch.matmul(weights.T, flat_data.float())  # [U, 5]


def merge_point_clouds(reference_points, detection_points):
    """obj_id 매칭을 통해 두 포인트 클라우드 병합
    
    Args:
        reference_points (Tensor): [N,5] 참조 포인트 (cam_id, obj_id, x,y,z)
        detection_points (Tensor): [M,5] 검출 포인트 (cam_id, obj_id, x,y,z)
    
    Returns:
        Tensor: [K,5] 병합된 포인트 클라우드 (K ≤ N)
    """
    # obj_id 정수형 변환
    ref_ids = reference_points[:, 1].long()
    det_ids = detection_points[:, 1].long()
    
    # 검출 포인트 딕셔너리 생성 (같은 obj_id가 여러 번 나타날 경우 마지막 항목 저장)
    det_dict = {det_ids[i].item(): detection_points[i] for i in range(len(det_ids))}
    
    # 병합 로직
    merged = []
    for idx, obj_id in enumerate(ref_ids):
        merged.append(det_dict.get(obj_id.item(), reference_points[idx]))
    
    return torch.stack(merged)

def differentiable_merge_point_clouds(reference_points, detection_points):
    """obj_id 매칭을 통해 두 포인트 클라우드 병합 (미분 가능 버전)
    Args:
        reference_points (Tensor): [N,5] (cam_id, obj_id, x,y,z)
        detection_points (Tensor): [M,5] (cam_id, obj_id, x,y,z)
    Returns:
        Tensor: [N,5] 병합된 포인트 클라우드
    """
    device = reference_points.device
    
    # 1. 객체 ID 추출 및 차원 확장
    ref_ids = reference_points[:, 1].unsqueeze(1)  # [N,1]
    det_ids = detection_points[:, 1].unsqueeze(0)  # [1,M]

    # 2. 매칭 마스크 생성 (객체 ID 일치 여부)
    matches = (ref_ids == det_ids).float()  # [N,M]

    # 3. 검출 포인트 확장 및 가중치 적용
    det_exp = detection_points.unsqueeze(0)  # [1,M,5]
    weighted_det = det_exp * matches.unsqueeze(-1)  # [N,M,5]

    # 4. 가중 합계 계산 (같은 객체 ID의 평균값)
    sum_det = weighted_det.sum(dim=1)  # [N,5]
    count = matches.sum(dim=1, keepdim=True)  # [N,1]
    count = torch.where(count == 0, torch.ones_like(count), count)
    avg_det = sum_det / count  # [N,5]

    # 5. 최종 병합 (매칭 없는 경우 원본 사용)
    mask = (matches.sum(dim=1) == 0).unsqueeze(1)  # [N,1]
    merged = torch.where(mask, reference_points, avg_det)

    return merged

# def differentiable_object_matching(pred_points, gt_points, obj_id_dim=1, temp=0.1):
#     """
#     미분 가능한 객체 ID 기반 포인트 클라우드 매칭
#     Args:
#         pred_points: [N, D] 예측 포인트 (obj_id = obj_id_dim)
#         gt_points: [M, D] GT 포인트 (obj_id = obj_id_dim)
#     Returns:
#         matched_pred: [K, D] 매칭된 예측 포인트
#         matched_gt: [K, D] 매칭된 GT 포인트
#     """
#     device = pred_points.device
    
#     # 1. 객체 ID 추출 (미분 가능한 정규화)
#     pred_ids = pred_points[:, obj_id_dim]  # [N]
#     gt_ids = gt_points[:, obj_id_dim]      # [M]

#     # 2. 유사도 행렬 계산 (미분 가능)
#     similarity = torch.sigmoid(10*(pred_ids.unsqueeze(1) - gt_ids.unsqueeze(0)))  # [N,M]

#     # 3. 양방향 최대 유사도 마스크 생성
#     pred_max = similarity.max(dim=1)[0]  # [N]
#     gt_max = similarity.max(dim=0)[0]    # [M]
    
#     # 4. 임계값 기반 마스킹 (soft thresholding)
#     pred_mask = torch.sigmoid(100*(pred_max - 0.5))  # [N]
#     gt_mask = torch.sigmoid(100*(gt_max - 0.5))      # [M]

#     # 5. 매칭된 포인트 선택 (미분 가능 샘플링)
#     matched_pred = pred_points * pred_mask.unsqueeze(1)  # [N,D]
#     matched_gt = gt_points * gt_mask.unsqueeze(1)        # [M,D]

#     # 6. 공통 포인트 수 일치 (패딩/마스킹)
#     k = min(int(pred_mask.sum()), int(gt_mask.sum()))
#     _, pred_topk = torch.topk(pred_mask, k)
#     _, gt_topk = torch.topk(gt_mask, k)

#     return matched_pred[pred_topk], matched_gt[gt_topk]

def differentiable_object_matching(pred_points, gt_points):
    # from geomloss import SamplesLoss
    gt_points = gt_points.float()
    
    # 1. 좌표 추출 (2D 텐서 보장)
    pred_coords = pred_points[..., :3].view(-1, 3)  # [N,3]
    gt_coords = gt_points[..., :3].view(-1, 3)      # [M,3]

    # 2. Sinkhorn 전송 계획 계산
    cost_matrix = torch.cdist(pred_coords, gt_coords, p=2)  # [N,M]
    transport_plan = F.softmax(-cost_matrix / 0.1, dim=1)   # [N,M]

    # 3. 미분 가능 매칭
    weights = torch.softmax(transport_plan * 10, dim=1)  # [N,M]
    matched_gt = torch.matmul(weights, gt_points)        # [N,D]

    return pred_points, matched_gt

def convert_to_bbox_coordinates_matched(
    rois_with_indices: torch.Tensor,  # [N, 6] (cam_id, obj_id, x1, y1, x2, y2)
    descale_corrs_pred_with_indices: torch.Tensor,  # [M, 5] (cam_id, obj_id, u, v, z)
    roi_size: tuple = (7, 7)
) -> torch.Tensor:
    """
    전체 이미지 좌표계 → ROI 정규화 좌표계 변환
    Returns: [M_valid, 3] (u_roi, v_roi, z)
    """
    device = rois_with_indices.device
    
    # 1. ROI 파라미터 사전 생성
    roi_dict = {}
    for roi in rois_with_indices:
        cam_id = int(roi[0].item())
        obj_id = int(roi[1].item())
        x1, y1, x2, y2 = roi[2:6]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # 수치 안정성을 위한 epsilon 추가
        if bbox_w < 1e-6 or bbox_h < 1e-6:
            continue
            
        scale_x = roi_size[0] / bbox_w
        scale_y = roi_size[1] / bbox_h
        roi_dict[(cam_id, obj_id)] = (x1, y1, scale_x, scale_y)

    # 2. 좌표 변환 수행
    matched_coords = []
    for corr in descale_corrs_pred_with_indices:
        cam_id = int(corr[0].item())
        obj_id = int(corr[1].item())
        key = (cam_id, obj_id)
        
        if key not in roi_dict:
            continue
            
        x1, y1, scale_x, scale_y = roi_dict[key]
        u_full = corr[2]
        v_full = corr[3]
        z = corr[4]
        
        # CORT 변환 공식 적용
        u_roi = (u_full - x1) * scale_x - 0.5  # -0.5는 중심 정렬을 위한 오프셋
        v_roi = (v_full - y1) * scale_y - 0.5
        
        matched_coords.append(torch.stack([u_roi, v_roi, z]))

    return torch.stack(matched_coords) if matched_coords else torch.empty((0,3), device=device)

   # ####### 검증용 corrs display ########
        # for camera_idx in range(6):
        #     mask = pts_hom_with_index[:, 0] == camera_idx
        #     lidar_points = pts_hom_with_index[mask, 1:5]
        #     veri_points_lidar2img_mis = (gt_KT[camera_idx] @ lidar_points.T).T
        #     veri_points_lidar2img_mis = torch.cat([veri_points_lidar2img_mis[:, :2] / veri_points_lidar2img_mis[:, 2:3], veri_points_lidar2img_mis[:, 2:3]], 1)

        #     lidar_points = det_xyz_hom_with_index[mask, 1:5]
        #     veri_points_lidar2img = (gt_KT[camera_idx] @ lidar_points.T).T
        #     veri_points_lidar2img = torch.cat([veri_points_lidar2img[:, :2] / veri_points_lidar2img[:, 2:3], veri_points_lidar2img[:, 2:3]], 1)
            
        #     veri_scaled_points_lidar2img_mis = scale_uvz_points(veri_points_lidar2img_mis)
        #     veri_scaled_points_lidar2img = scale_uvz_points(veri_points_lidar2img)
        #     veri_normal_points_lidar2img_mis = veri_scaled_points_lidar2img_mis.clone()
        #     veri_normal_points_lidar2img = veri_scaled_points_lidar2img.clone()

        #     veri_normal_points_lidar2img_mis[:, 0] = veri_normal_points_lidar2img_mis[:, 0]/640
        #     veri_normal_points_lidar2img_mis[:, 1] = veri_normal_points_lidar2img_mis[:, 1]/192 
        #     if veri_normal_points_lidar2img_mis[:, 2].numel() > 0:
        #         veri_normal_points_lidar2img_mis[:, 2] = (veri_normal_points_lidar2img_mis[:, 2]-torch.min(veri_normal_points_lidar2img_mis[:, 2]))\
        #             /(torch.max(veri_normal_points_lidar2img_mis[:, 2]) - torch.min(veri_normal_points_lidar2img_mis[:, 2]))
        #     else :
        #         veri_normal_points_lidar2img_mis[:, 2] = (veri_normal_points_lidar2img_mis[:, 2]-0)/(80 - 0)
        #     veri_normal_points_lidar2img_mis[:, 0] += 0.5
            
        #     veri_normal_points_lidar2img[:, 0] = veri_normal_points_lidar2img[:, 0]/640
        #     veri_normal_points_lidar2img[:, 1] = veri_normal_points_lidar2img[:, 1]/192
        #     if veri_normal_points_lidar2img[:, 2].numel() > 0:
        #         veri_normal_points_lidar2img[:, 2] = (veri_normal_points_lidar2img[:, 2]-torch.min(veri_normal_points_lidar2img[:, 2]))\
        #             /(torch.max(veri_normal_points_lidar2img[:, 2]) - torch.min(veri_normal_points_lidar2img[:, 2]))
        #     else :
        #         veri_normal_points_lidar2img[:, 2] = (veri_normal_points_lidar2img[:, 2]-0)/(80 - 0)
            
        #     #### corrspondence points display ######
        #     import matplotlib.pyplot as plt
        #     # 입력 이미지 처리
        #     img_tensor = sbs_img[camera_idx]  # [3, 192, 1280]
        #     # denorm_img = img_tensor / 2 + 0.5  # 정규화 해제
        #     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        #     # 3차원 좌표에서 2D 이미지 좌표 추출 (z값 제거)
        #     left_pts = veri_scaled_points_lidar2img.cpu().numpy()[:, :2]  # [N,2] (u,v)
        #     right_pts = veri_scaled_points_lidar2img_mis.cpu().numpy()[:, :2]  # [N,2]
        #     right_pts[:, 0] += 640 

        #     # 좌표 형상 보정
        #     left_pts = left_pts.reshape(-1, 2)  # [N,2] 보장
        #     right_pts = right_pts.reshape(-1, 2)

        #     # 좌표 범위 클리핑
        #     H, W = img_np.shape[:2]
        #     left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W-1)
        #     left_pts[:, 1] = np.clip(left_pts[:, 1], 0, H-1)
        #     right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W-1)
        #     right_pts[:, 1] = np.clip(right_pts[:, 1], 0, H-1)

        #     # NaN 값 필터링
        #     valid_mask = ~(np.isnan(left_pts).any(axis=1) | np.isnan(right_pts).any(axis=1))
        #     left_pts = left_pts[valid_mask]
        #     right_pts = right_pts[valid_mask]

        #     # 시각화
        #     plt.figure(figsize=(20, 6))
        #     plt.imshow(img_np)

        #     # 포인트 및 연결선 플롯
        #     plt.scatter(left_pts[:,0], left_pts[:,1], 
        #                 c='cyan', s=5, edgecolors='k', linewidths=0.8, label='Left Points')
        #     plt.scatter(right_pts[:,0], right_pts[:,1], 
        #                 c='magenta', s=5, edgecolors='k', linewidths=0.8, label='Right Points')

        #     for left_p, right_p in zip(left_pts, right_pts):
        #         plt.plot([left_p[0], right_p[0]], [left_p[1], right_p[1]],
        #                 color='yellow', linestyle='--', linewidth=1.5, alpha=0.6)

        #     plt.axis('off')
        #     plt.legend(loc='upper right', prop={'size': 12})
        #     plt.savefig('correspond.jpg', dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print ("end")

def transform_uv_points(rois_with_indices, uv_set):
    device = rois_with_indices.device
    num_rois = rois_with_indices.size(0)
    
    cam_ids = rois_with_indices[:, 0].long()
    obj_ids = rois_with_indices[:, 1].long()
    x_min, y_min = rois_with_indices[:, 2], rois_with_indices[:, 3]
    x_max, y_max = rois_with_indices[:, 4], rois_with_indices[:, 5]
    
    b_box_cx = (x_min + x_max) / 2.0
    b_box_cy = (y_min + y_max) / 2.0

    transformed_points = torch.zeros((num_rois, 8), device=device)

    for i in range(num_rois):
        cam_id = cam_ids[i]
        obj_id = obj_ids[i]
        target_cx = b_box_cx[i]
        target_cy = b_box_cy[i]

        camera_uv = uv_set[cam_id]
        
        u_diff = camera_uv[:, 0] - target_cx
        v_diff = camera_uv[:, 1] - target_cy
        distances = u_diff.pow(2) + v_diff.pow(2)
        min_idx = torch.argmin(distances)

        original_u = camera_uv[min_idx, 0]
        original_v = camera_uv[min_idx, 1]
        
        # 변위 계산 (target으로 이동하기 위한)
        delta_u = target_cx - original_u
        delta_v = target_cy - original_v

        # 변환 적용
        transformed_u = original_u + delta_u  # = target_cx
        transformed_v = original_v + delta_v  # = target_cy
        transformed_u_prime = camera_uv[min_idx, 2] + delta_u
        transformed_v_prime = camera_uv[min_idx, 3] + delta_v

        transformed_points[i] = torch.tensor([
            cam_id, obj_id,
            target_cx, target_cy,
            transformed_u, transformed_v,
            transformed_u_prime, transformed_v_prime
        ], device=device)

    return transformed_points

def get_center_points(rois_with_indices):
    """
    Args:
        rois_with_indices: Tensor of shape [num_obj, 6] (cam_id, obj_id, x_min, y_min, x_max, y_max)
    Returns:
        center_points: Tensor of shape [num_obj, 4] (cam_id, obj_id, center_x, center_y)
    """
    cam_ids = rois_with_indices[:, 0]
    obj_ids = rois_with_indices[:, 1]
    x_min = rois_with_indices[:, 2]
    y_min = rois_with_indices[:, 3]
    x_max = rois_with_indices[:, 4]
    y_max = rois_with_indices[:, 5]

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    center_points = torch.stack([cam_ids, obj_ids, center_x, center_y], dim=1)
    return center_points

# def batch_rois_center_by_cam_id(rois_center, num_cams=6, batch_size=100):
#     device = rois_center.device
#     batched_centers = torch.zeros((num_cams, batch_size, 4), device=device)  # [cam_id, obj_id, center_x, center_y]
    
#     for cam_id in range(num_cams):
#         cam_mask = (rois_center[:, 0] == cam_id)
#         cam_centers = rois_center[cam_mask]  # [n, 4] (cam_id, obj_id, cx, cy)
        
#         n = cam_centers.size(0)
        
#         if n == 0:
#             continue
#         elif n < batch_size:
#             repeat_factor = (batch_size + n - 1) // n
#             cam_centers = cam_centers.repeat(repeat_factor, 1)[:batch_size]
#         else:
#             cam_centers = cam_centers[:batch_size]
        
#         # 모든 정보 보존
#         batched_centers[cam_id, :, :] = cam_centers
    
#     return batched_centers


def batch_rois_center_by_cam_id(rois_center, num_cams=6, batch_size=100):
    device = rois_center.device
    batched_centers = torch.zeros((num_cams, batch_size, 4), device=device)
    
    # 원본 객체 ID 저장
    original_obj_ids = rois_center[:, 1].cpu().numpy()
    
    for cam_id in range(num_cams):
        cam_mask = (rois_center[:, 0] == cam_id)
        cam_centers = rois_center[cam_mask]
        n = cam_centers.size(0)
        
        if n == 0:
            continue
            
        # 고유한 객체 ID 추출
        obj_ids = cam_centers[:, 1].cpu().numpy()
        unique_obj_ids = np.unique(obj_ids)
        num_unique_objs = len(unique_obj_ids)
        
        if num_unique_objs <= batch_size:
            selected_indices = torch.arange(n, device=device)
            
            if n < batch_size:
                repeat_factor = (batch_size + n - 1) // n
                cam_centers = cam_centers.repeat(repeat_factor, 1)[:batch_size]
            else:
                pass
        else:
            selected_indices = []
            for obj_id in unique_obj_ids:
                obj_indices = np.where(obj_ids == obj_id)[0]
                selected_idx = np.random.choice(obj_indices)
                selected_indices.append(selected_idx)
                
            if len(selected_indices) < batch_size:
                remaining = batch_size - len(selected_indices)
                extra_indices = np.random.choice(
                    np.setdiff1d(np.arange(n), selected_indices),
                    size=remaining,
                    replace=False
                )
                selected_indices.extend(extra_indices)
                
            selected_indices = torch.tensor(selected_indices, device=device)
            cam_centers = cam_centers[selected_indices]
        
        batched_centers[cam_id, :cam_centers.size(0)] = cam_centers[:batch_size]
    
    # ===== 객체 ID 검증 코드 =====
    # 1. 원본과 배치된 객체 ID 추출
    batched_obj_ids = batched_centers[:, :, 1].flatten().cpu().numpy()
    
    # 2. 값 동일 여부 검증
    original_set = set(original_obj_ids)
    batched_set = set(batched_obj_ids)
    
    # 0 값 제외 (패딩 값)
    original_set_nonzero = original_set - {0}
    batched_set_nonzero = batched_set - {0}
    
    # 3. 연속성 검증 함수
    def check_continuity(arr):
        if len(arr) == 0:
            return True
        sorted_arr = np.sort(arr)
        return np.all(np.diff(sorted_arr) == 1)
    
    # 4. 원본 객체 ID 연속성 검증
    original_continuous = check_continuity(np.array(list(original_set_nonzero)))
    
    # 5. 배치된 객체 ID 연속성 검증
    batched_continuous = check_continuity(np.array(list(batched_set_nonzero)))
    
    # 6. 값 동일 여부 검증
    same_values = original_set_nonzero == batched_set_nonzero
    
    # # 7. 결과 출력
    # print(f"[객체 ID 검증]")
    # print(f"원본 객체 수: {len(original_set_nonzero)}, 배치된 객체 수: {len(batched_set_nonzero)}")
    # print(f"원본 연속성: {original_continuous}, 배치 연속성: {batched_continuous}")
    # print(f"객체 ID 동일 여부: {same_values}")
    
    # if not same_values:
    #     missing = original_set_nonzero - batched_set_nonzero
    #     extra = batched_set_nonzero - original_set_nonzero
    #     print(f"누락된 객체 ID: {missing}")
    #     print(f"추가된 객체 ID: {extra}")
    #     print("누락되었습니다")
    
    # if not batched_continuous:
    #     batched_sorted = np.sort(list(batched_set_nonzero))
    #     gaps = np.where(np.diff(batched_sorted) > 1)[0]
    #     print(f"배치 객체 ID 불연속 지점: {batched_sorted[gaps]}")
    
    return batched_centers

def remove_duplicate_objs(corrs_pred_with_obj):
    """
    객체 ID 기준 중복 제거 및 결과 포맷 변환
    - PyTorch 버전 호환성 해결
    - 객체 ID 연속성 체크 추가
    - 텐서 크기 불일치 해결
    
    Args:
        corrs_pred_with_obj: [num_cams, batch_size, 3] 텐서 
            (obj_id, center_pred_x, center_pred_y)
            
    Returns:
        [number_of_unique_obj, 4] 텐서 
        (cam_id, obj_id, center_pred_x, center_pred_y)
    """
    num_cams, batch_size, _ = corrs_pred_with_obj.shape
    
    # 1. 카메라 ID 텐서 생성
    cam_ids = torch.arange(num_cams, device=corrs_pred_with_obj.device)
    cam_ids = cam_ids.view(-1, 1, 1).expand(-1, batch_size, 1)
    
    # 2. 모든 정보 결합 [cam_id, obj_id, pred_x, pred_y]
    combined = torch.cat([cam_ids.float(), corrs_pred_with_obj], dim=-1)
    
    # 3. 배치 차원 병합 [num_cams * batch_size, 4]
    flat_combined = combined.view(-1, 4)
    
    # 4. 객체 ID 추출 및 연속성 체크
    obj_ids = flat_combined[:, 1]
    unique_ids, counts = torch.unique(obj_ids, return_counts=True)
    
    # 5. 객체 ID 연속성 검증
    if not torch.all(torch.diff(unique_ids) == 1):
        print("경고: 객체 ID가 연속적이지 않음. 누락된 객체 존재 가능")
    
    # 6. 중복 제거 (첫 번째 발생만 유지)
    _, unique_indices = torch.unique(obj_ids, return_inverse=True)
    first_occurrence = torch.zeros_like(obj_ids, dtype=torch.bool)
    
    for obj_id in unique_ids:
        indices = (obj_ids == obj_id).nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            first_occurrence[indices[0]] = True
    
    # 7. 고유 객체 선택
    unique_objs = flat_combined[first_occurrence]
    
    # 8. 크기 검증
    if unique_objs.size(0) != unique_ids.size(0):
        print(f"크기 불일치: 고유 객체 {unique_ids.size(0)}개, 결과 {unique_objs.size(0)}개")
    
    return unique_objs

# def remove_duplicate_objs(corrs_pred_with_obj):
#     """
#     객체 ID 기준 중복 제거 및 결과 포맷 변환 (z값 포함)
#     Args:
#         corrs_pred_with_obj: [num_cams, batch_size, 4] 텐서 
#             (obj_id, center_pred_x, center_pred_y, center_pred_z)
#     Returns:
#         [number_of_unique_obj, 5] 텐서 
#         (cam_id, obj_id, center_pred_x, center_pred_y, center_pred_z)
#     """
#     num_cams, batch_size, _ = corrs_pred_with_obj.shape

#     # 1. 카메라 ID 텐서 생성
#     cam_ids = torch.arange(num_cams, device=corrs_pred_with_obj.device)
#     cam_ids = cam_ids.view(-1, 1, 1).expand(-1, batch_size, 1)  # [num_cams, batch_size, 1]

#     # 2. 모든 정보 결합 [cam_id, obj_id, x, y, z]
#     combined = torch.cat([cam_ids.float(), corrs_pred_with_obj], dim=-1)  # [num_cams, batch_size, 5]

#     # 3. 배치 차원 병합 [num_cams * batch_size, 5]
#     flat_combined = combined.reshape(-1, 5)

#     # 4. obj_id 기준 첫 등장 인덱스만 유지
#     obj_ids = flat_combined[:, 1]
#     # unique obj_id의 첫 등장 인덱스만 추출 (PyTorch 공식 우회법)
#     unique_obj_ids, inverse_indices = torch.unique(obj_ids, sorted=True, return_inverse=True)
#     perm = torch.arange(obj_ids.size(0), dtype=inverse_indices.dtype, device=obj_ids.device)
#     inverse_indices, perm = inverse_indices.flip(0), perm.flip(0)
#     first_indices = inverse_indices.new_empty(unique_obj_ids.size(0)).scatter_(0, inverse_indices, perm)
#     unique_objs = flat_combined[first_indices]

#     return unique_objs


def denormalize_uv_points(normalized_points, scaled_size=(192, 1280)):
    """
    정규화된 UV 좌표를 스케일링된 이미지 픽셀 좌표로 복원
    Args:
        normalized_points: [N, 2] 정규화된 UV 좌표 (0~1 범위)
        scaled_size: (H_scaled, W_scaled) 스케일링된 이미지 크기
    Returns:
        [N, 2] 스케일링된 이미지 픽셀 좌표
    """
    H_scaled, W_scaled = scaled_size
    denormalized = normalized_points.clone()
    denormalized[..., 2] *= W_scaled  # u 좌표 복원 (0~1 → 0~W_scaled)
    denormalized[..., 3] *= H_scaled  # v 좌표 복원 (0~1 → 0~H_scaled)
    return denormalized

def descale_uvz_points(uvz_tensor, original_size=(192,640),target_size=(900,1600)):
    """
    UVZ 좌표를 이미지 스케일링 비율에 맞게 변환
    Args:
        uvz_tensor: (N, 3) 형태의 텐서 [u, v, z]
        original_size: (H, W) 원본 이미지 크기
        target_size: (H, W) 타겟 이미지 크기
    Returns:
        스케일링된 (N, 3) UVZ 텐서
    """
    # 스케일링 계수 계산 (높이, 너비)
    scale_h = target_size[0] / original_size[0]  
    scale_w = target_size[1] / original_size[1] 
    
    # UV 좌표 스케일링 (깊이 z는 변경 없음)
    scaled_uvz = uvz_tensor.clone()
    scaled_uvz[..., 2] = scaled_uvz[..., 2] - original_size[1]
    scaled_uvz[..., 2] *= scale_w  # u 좌표(너비 방향) 스케일
    scaled_uvz[..., 3] *= scale_h  # v 좌표(높이 방향) 스케일
    
    return scaled_uvz

def save_img_comparison_as_jpg(img, warped_img, cam_idx=0, save_path='comparison.jpg'):
    img_np = img[cam_idx].cpu().permute(1, 2, 0).numpy()
    warped_np = warped_img[cam_idx].cpu().permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(img_np)
    axs[0].set_title(f'Original Image - Camera {cam_idx}')
    axs[0].axis('off')

    axs[1].imshow(warped_np)
    axs[1].set_title(f'Warped Image - Camera {cam_idx}')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def bbox2roi_with_camidx(bbox_list):
    """
    Args:
        bbox_list (list[Tensor]): 각 박스 tensor에서 첫 번째 컬럼이 camera index로 들어 있음
    Returns:
        Tensor: shape (n, 6), [batch_ind, camera_idx, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            cam_indices = bboxes[:, 0].unsqueeze(1)              # 첫 번째 값(camera index)
            coords = bboxes[:, 1:5]                              # 박스 좌표 [x1, y1, x2, y2]
            rois = torch.cat([img_inds, cam_indices, coords], dim=-1)  # [N, 6]
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

def save_verification_image(
    original_img, 
    augmented_img, 
    original_preds_tensor, 
    augmented_preds_tensor, 
    save_path,
    score_thr=0.3
):
    """
    원본/증강 이미지에 BBox를 그린 검증 결과를 단일 이미지 파일로 저장합니다.

    Args:
        original_img (np.ndarray): 원본 이미지 (H, W, 3).
        augmented_img (np.ndarray): 증강된 이미지 (fH, fW, 3).
        original_preds_tensor (torch.Tensor): 원본 좌표계의 예측 텐서 (N, 6).
        augmented_preds_tensor (torch.Tensor): 증강 좌표계의 예측 텐서 (N, 6).
        save_path (str): 이미지를 저장할 경로 (e.g., 'output/verification.png').
        score_thr (float, optional): 지정된 점수 이상의 BBox만 표시.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 예측 텐서에서 정보 분리
    orig_boxes, orig_scores, orig_labels = original_preds_tensor[:, :4], original_preds_tensor[:, 4], original_preds_tensor[:, 5]
    aug_boxes, aug_scores, aug_labels = augmented_preds_tensor[:, :4], augmented_preds_tensor[:, 4], augmented_preds_tensor[:, 5]

    # 1. 원본 이미지에 BBox 그리기
    vis_orig = original_img.copy()
    for i, bbox in enumerate(orig_boxes):
        if orig_scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = bbox.int().cpu().numpy()
        cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'L:{int(orig_labels[i].item())} | {orig_scores[i].item():.2f}'
        cv2.putText(vis_orig, text, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

    # 2. 증강 이미지에 BBox 그리기
    vis_aug = augmented_img.copy().astype(np.uint8)
    for i, bbox in enumerate(aug_boxes):
        if aug_scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = bbox.int().cpu().numpy()
        cv2.rectangle(vis_aug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f'L:{int(aug_labels[i].item())} | {aug_scores[i].item():.2f}'
        cv2.putText(vis_aug, text, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

    # 3. Matplotlib으로 두 이미지 비교 그림 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image with Converted BBoxes (Green)')
    ax1.axis('off')

    ax2.imshow(vis_aug)
    ax2.set_title('Augmented Image with Predicted BBoxes (Red)')
    ax2.axis('off')

    plt.tight_layout()

    # 4. 그림을 파일로 저장하고 plot을 닫음
    # 저장 경로의 디렉터리가 없으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig) # 메모리 해제를 위해 꼭 닫아주세요.
    print(f"Verification image saved to: {save_path}")

def denormalize_image(tensor, mean, std):
    """
    정규화된 이미지 텐서를 시각화를 위한 uint8 NumPy 배열로 역정규화합니다.
    """
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)

    # 역정규화: (정규화된 값 * std) + mean
    denorm_tensor = tensor * std + mean

    # (C, H, W) -> (H, W, C) 변환 및 NumPy 배열로 변경
    img_np = denorm_tensor.permute(1, 2, 0).cpu().numpy()

    # 0-255 범위로 클리핑하고 uint8 타입으로 변환
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    return img_np

def save_batch_predictions_to_file(
    batch_inputs_dict, 
    reshaped_data_samples, 
    augmented_preds_list, 
    original_preds_list,
    current_step,
    save_dir='vis_results',
    view_index=0, 
    score_thr=0.3
):
    """
    배치 데이터 중 특정 뷰를 선택하여 검증 이미지를 파일로 저장합니다.
    """
    # ====================================================================
    # ===== ⚠️ 중요: 사용하시는 데이터셋의 정규화 값을 입력하세요. =====
    # ===== (MMDetection3D config 파일의 'img_norm_cfg'에 있습니다)
    # ====================================================================
    IMG_MEAN = [123.675, 116.28, 103.53] # ImageNet 기본값 예시
    IMG_STD = [58.395, 57.12, 57.375]   # ImageNet 기본값 예시
    
    num_cameras = 6
    batch_idx = view_index // num_cameras
    cam_idx = view_index % num_cameras
    
    # --- 원본 이미지 처리 ---
    original_imgs_batch = batch_inputs_dict.get('img_original')
    if original_imgs_batch is None:
        print("저장 실패: batch_inputs_dict에 'img_original'이 없습니다.")
        return
    original_img_tensor = original_imgs_batch[batch_idx][cam_idx]
    # 값의 범위가 0-255이므로 uint8로 타입만 변환합니다.
    original_img = original_img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # --- 증강 이미지 처리 ---
    augmented_imgs_batch = batch_inputs_dict.get('imgs')
    if augmented_imgs_batch is None:
        print("저장 실패: batch_inputs_dict에 'imgs'가 없습니다.")
        return
    augmented_img_tensor = augmented_imgs_batch[batch_idx][cam_idx]
    # 정규화되어 있으므로, denormalize_image 함수를 사용해 역정규화합니다.
    augmented_img = denormalize_image(augmented_img_tensor, IMG_MEAN, IMG_STD)
    
    # 예측 데이터 선택 및 저장 경로 생성
    aug_preds_tensor = augmented_preds_list[view_index]
    orig_preds_tensor = original_preds_list[view_index]
    save_path = os.path.join(save_dir, f'step_{current_step}_view_{view_index}.png')

    # 이미지 저장 함수 호출
    save_verification_image(
        original_img,
        augmented_img,
        orig_preds_tensor,
        aug_preds_tensor,
        save_path=save_path,
        score_thr=score_thr
    )

def visualize_bev_proposals(det_xyz, gt_bboxes_3d, step, save_path="bev_visualization.png"):
    """
    카메라 제안(det_xyz)과 정답 박스를 BEV 평면에 시각화합니다.

    Args:
        det_xyz (torch.Tensor): [num_proposals, 3] 모양의 카메라 제안 포인트.
        gt_bboxes_3d (Box3DMode): Ground truth 3D 바운딩 박스 객체.
        step (int): 현재 학습 스텝 (파일 이름에 사용).
        save_path (str): 이미지를 저장할 경로.
    """
    # 텐서를 CPU의 NumPy 배열로 변환
    points = det_xyz.detach().cpu().numpy()
    gt_boxes = gt_bboxes_3d.tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black') # 배경을 검은색으로

    # 1. 카메라 제안(det_xyz)을 스캐터 플롯으로 그립니다.
    # Z 좌표(높이)를 색상으로 사용합니다.
    scatter = ax.scatter(
        points[:, 0],  # X 좌표
        points[:, 1],  # Y 좌표
        c=points[:, 2],  # Z 좌표를 색상으로 매핑
        cmap='viridis',  # 색상 맵 (viridis, jet 등)
        s=1,             # 점 크기
        alpha=0.7,       # 투명도
        label='Camera Proposals'
    )
    
    # 색상 막대(Colorbar)를 추가하여 Z값의 의미를 표시
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label('Z coordinate (Height)')

    # 2. Ground Truth 바운딩 박스를 그립니다. (녹색)
    for i in range(len(gt_boxes)):
        # MMDetection3D의 Box 객체는 보통 .corners 속성을 가집니다.
        # BEV 평면의 코너 (앞 왼쪽, 뒤 왼쪽, 뒤 오른쪽, 앞 오른쪽)
        corners = gt_bboxes_3d[i].corners.cpu().numpy()
        # ================================================================
        # # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 디버깅용 임시 코드 시작 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # # 목적: 8개 코너의 위치와 번호(0~7)를 직접 그려서 확인합니다.
        
        # # 1. 8개 코너의 X, Y 좌표를 모두 가져옵니다.
        # bev_all_8_corners = corners[0, :, :2]
        
        # # 2. 8개 코너를 알아보기 쉽게 빨간 점으로 그립니다.
        # ax.scatter(bev_all_8_corners[:, 0], bev_all_8_corners[:, 1], c='red', s=20, zorder=3)
        
        # # 3. 각 빨간 점 옆에 인덱스 번호(0~7)를 흰색 텍스트로 표시합니다.
        # for j in range(8):
        #     ax.text(bev_all_8_corners[j, 0] + 0.5, bev_all_8_corners[j, 1] + 0.5, str(j), color='white', fontsize=12)
            
        # # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 디버깅용 임시 코드 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # ================================================================
        bev_corners = corners[0, [0, 1, 2, 3], :2]
        
        # Polygon을 사용하여 회전된 사각형을 그립니다.
        polygon = patches.Polygon(
            bev_corners, 
            edgecolor='lime', # 라임색
            facecolor='none', 
            linewidth=2,
            label='Ground Truth' if i == 0 else "" # 범례 중복 방지
        )
        ax.add_patch(polygon)
        
    # 3. 자차(Ego Vehicle) 위치를 표시합니다. (흰색)
    ego_vehicle = patches.Rectangle(
        (-1.0, -2.5), 2.0, 5.0, # 대략적인 차량 크기
        edgecolor='white', 
        facecolor='none', 
        linewidth=2,
        label='Ego Vehicle'
    )
    ax.add_patch(ego_vehicle)
    
    # 4. 플롯 설정
    ax.set_xlim(-51.2, 51.2) # nuScenes의 일반적인 BEV 범위
    ax.set_ylim(-51.2, 51.2)
    ax.set_aspect('equal', adjustable='box') # 가로세로 비율을 1:1로
    ax.set_title(f"BEV Camera Proposals @ Step {step}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    
    # 범례 핸들 중복 제거
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # 이미지 저장 및 닫기
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ BEV visualization saved to {save_path}")

def visualize_full_pipeline(
    cam_proposals_xyz, cam_proposals_feat,  # <-- 추가된 인자
    query_pos, lidar_only_feat, fused_feat, final_feat, 
    gt_bboxes_3d, pc_range, voxel_size, step, save_path
):
    """
    카메라 퓨전 전/후의 BEV 쿼리 특징을 시각화합니다.
    """
    # 텐서를 CPU의 NumPy 배열로 변환
    cam_xyz = cam_proposals_xyz.detach().cpu().numpy()
    cam_feat = cam_proposals_feat.detach().cpu().numpy()
    pos = query_pos.detach().cpu().numpy()
    feat1 = lidar_only_feat.detach().cpu().numpy()
    feat2 = fused_feat.detach().cpu().numpy()
    feat3 = final_feat.detach().cpu().numpy()
    gt_boxes = gt_bboxes_3d.tensor.cpu().numpy()

    # --- ✨✨✨ START: FIX - 카메라 제안 좌표 역정규화 ✨✨✨ ---
    # 입력된 cam_xyz는 [0, 1]로 정규화된 상태입니다.
    # 이를 pc_range를 이용해 다시 실제 미터(m) 단위 좌표로 변환합니다.
    cam_metric_x = cam_xyz[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    cam_metric_y = cam_xyz[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    # Z 좌표는 색상으로만 사용되므로 변환이 필수는 아니지만, 일관성을 위해 변환할 수 있습니다.
    # cam_metric_z = cam_xyz[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    # --- ✨✨✨ END: FIX --- ✨✨✨

    # 각 특징 벡터의 L2 Norm(크기)을 계산
    cam_norms = np.linalg.norm(cam_feat, axis=-1)
    lidar_norms = np.linalg.norm(feat1, axis=-1)
    fused_norms = np.linalg.norm(feat2, axis=-1)
    final_norms = np.linalg.norm(feat3, axis=-1)

    all_norms = np.concatenate([cam_norms, lidar_norms, fused_norms, final_norms])
    vmin = all_norms.min()
    vmax = all_norms.max()

    # ✨ FIX: 1x4 서브플롯 생성
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(48, 12))
    fig.suptitle(f"BEV Feature Pipeline @ Step {step}", fontsize=20)

    # --- Plot 0: 초기 (Camera-Only Proposals) ---
    ax0.set_title("0. Initial (Camera-Only)")
    ax0.set_facecolor('black')
    # ✨ FIX: 역정규화된 미터 단위 좌표를 사용하여 그립니다.
    scatter0 = ax0.scatter(
        cam_metric_x, 
        cam_metric_y, 
        c=cam_norms, 
        cmap='viridis', 
        s=5, 
        alpha=0.7, 
        vmin=vmin, 
        vmax=vmax
    )
    fig.colorbar(scatter0, ax=ax0, label='Feature Norm (Strength)')

    # --- ✨✨✨ START: FIX - 시각화 편향 보정 ✨✨✨ ---
    # '좌하향' 편향(e.g., 피처가 -0.5 셀만큼 밀림)을 보정하기 위해
    # 쿼리 위치(pos)에 0.5 그리드 셀만큼의 오프셋을 더해줍니다.
    pos_corrected = pos
    # --- ✨✨✨ END: FIX --- ✨✨✨

    # --- Plot 1: 초기 (LiDAR-Only Queries) ---
    ax1.set_title("1. Initial (LiDAR-Only)")
    ax1.set_facecolor('black')
    out_size_factor = 4 # Transfusion의 경우
    metric_x = pos_corrected[:, 0] * voxel_size[0] * out_size_factor + pc_range[0]
    metric_y = pos_corrected[:, 1] * voxel_size[1] * out_size_factor + pc_range[1]
    scatter1 = ax1.scatter(metric_x, metric_y, c=lidar_norms, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax)
    fig.colorbar(scatter1, ax=ax1, label='Feature Norm (Strength)')

    # --- Plot 2: 중간 (After Camera Fusion) ---
    ax2.set_title("2. Mid-Fusion (LiDAR + Camera)")
    ax2.set_facecolor('black')
    scatter2 = ax2.scatter(metric_x, metric_y, c=fused_norms, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax)
    fig.colorbar(scatter2, ax=ax2, label='Feature Norm (Strength)')
    
    # --- Plot 3: 최종 (After Decoder Refinement) ---
    ax3.set_title("3. Final (Refined)")
    ax3.set_facecolor('black')
    scatter3 = ax3.scatter(metric_x, metric_y, c=final_norms, cmap='viridis', s=15, alpha=0.8, vmin=vmin, vmax=vmax)
    fig.colorbar(scatter3, ax=ax3, label='Feature Norm (Strength)')

    # ✨ FIX: 모든 플롯에 공통 요소 그리기
    for ax in [ax0, ax1, ax2, ax3]:
        # ... (GT Box, Ego Vehicle, 축 설정 등 공통 로직은 이전과 동일) ...
        for i in range(len(gt_boxes)):
            corners = gt_bboxes_3d[i].corners.cpu().numpy()
            bev_corners = corners[0, [0, 1, 2, 3], :2]
            polygon = patches.Polygon(bev_corners, edgecolor='lime', facecolor='none', linewidth=2,
                                      label='Ground Truth' if i == 0 else "")
            ax.add_patch(polygon)
        
        ego_vehicle = patches.Rectangle((-1.0, -2.5), 2.0, 5.0, edgecolor='white', facecolor='none',
                                        linewidth=2, label='Ego Vehicle')
        ax.add_patch(ego_vehicle)
        
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Full pipeline visualization saved to {save_path}")

# --- ✨ START: Calibration Visualization Helpers ✨ ---

def project_points_to_image(points_lidar_frame, lidar_to_img_proj_matrix, H, W):
    """
    LiDAR 좌표계의 3D 포인트를 이미지 평면에 투영하고 필터링합니다.

    Args:
        points_lidar_frame (torch.Tensor): LiDAR 좌표계의 포인트 [N, 3+].
        lidar_to_img_proj_matrix (torch.Tensor): 3x4 LiDAR-to-Image 투영 행렬 (K @ T_lidar2cam).
        H (int): 이미지 높이.
        W (int): 이미지 너비.

    Returns:
        torch.Tensor: 이미지 평면 상의 유효한 포인트 [M, 2] (u, v 좌표).
                     None 반환 시 유효 포인트 없음.
    """
    if points_lidar_frame is None or lidar_to_img_proj_matrix is None:
        return None
        
    points_xyz = points_lidar_frame[:, :3]
    points_ones = torch.ones_like(points_xyz[:, :1])
    points_lidar_hom = torch.cat([points_xyz, points_ones], dim=-1) # [N, 4]

    # points_img_hom: [N, 3] = lidar_to_img_proj_matrix @ points_lidar_hom.T
    points_img_hom = (lidar_to_img_proj_matrix @ points_lidar_hom.T).T

    # Perspective division
    depth = points_img_hom[:, 2:3]
    points_uv = points_img_hom[:, :2] / (depth + 1e-8) # [N, 2]

    # Filter points behind camera or outside image boundaries
    mask_depth = (depth.squeeze() > 0)
    mask_u = (points_uv[:, 0] >= 0) & (points_uv[:, 0] < W)
    mask_v = (points_uv[:, 1] >= 0) & (points_uv[:, 1] < H)
    valid_mask = mask_depth & mask_u & mask_v

    if valid_mask.any():
        return points_uv[valid_mask]
    else:
        return None

def visualize_calibration_effect(img, pts_gt, pts_broken, pts_corr, step, save_path):
    """
    카메라 이미지 위에 GT, Broken, Corrected LiDAR 포인트를 시각화합니다.
    """
    plt.figure(figsize=(16, 9))
    plt.imshow(img)

    # 점 크기 및 투명도 설정
    s = 1 # 점 크기
    alpha = 0.6 # 투명도

    if pts_broken is not None:
        pts_broken_np = pts_broken.cpu().numpy()
        plt.scatter(pts_broken_np[:, 0], pts_broken_np[:, 1], color='red', s=s, alpha=alpha, label='Broken Calibration')

    if pts_corr is not None:
        pts_corr_np = pts_corr.cpu().numpy()
        plt.scatter(pts_corr_np[:, 0], pts_corr_np[:, 1], color='cyan', s=s, alpha=alpha, label='Corrected by Model')

    if pts_gt is not None:
        pts_gt_np = pts_gt.cpu().numpy()
        plt.scatter(pts_gt_np[:, 0], pts_gt_np[:, 1], color='lime', s=s, alpha=alpha, label='Ground Truth Calibration')

    plt.title(f"Calibration Check @ Step {step}")
    plt.legend(markerscale=10) # 범례의 점 크기를 키움
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Calibration visualization saved to {save_path}")

# --- ✨ END: Calibration Visualization Helpers ✨ ---