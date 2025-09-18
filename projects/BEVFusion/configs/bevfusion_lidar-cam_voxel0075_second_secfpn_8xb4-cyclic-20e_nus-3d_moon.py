_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 80.0, 0.5], # original
    # 'depth': [1.0, 60.0, 0.5],
}
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion.my_collate',
             'projects.BEVFusion.bevfusion.cotr'],
    allow_failed_imports=False)


model = dict(
    type='BEVFusion',
    class_names=class_names,
    train_cfg=dict(
        complement_2d_gt=0.35, # <-- 커스텀 설정을 여기로 이동
        # 레퍼런스 코드에서 사용했던 다른 설정도 여기에 추가
        detection_proposal=dict(min_bbox_size=0) 
    ),
    data_preprocessor=dict(
        type='CustomDet3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    # --- 2. 새로 추가할 2D Detection Head ---
    img_bbox_head=dict(
        type='mmdet.RetinaHead',
        num_classes=10,  # TODO: 데이터셋에 맞는 2D 클래스 개수로 수정 (예: nuImages는 10개)
        in_channels=256, # img_neck의 out_channels와 동일해야 함
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            # strides=[8, 16, 32] # FPN 각 레벨에 대응하는 stride
            strides=[8, 16] # FPN 각 레벨에 대응하는 stride
        ),
        bbox_coder=dict(
            type='mmdet.DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            sampler=dict(
                type='mmdet.PseudoSampler'),  # RetinaNet은 모든 앵커를 사용하므로 PseudoSampler 사용
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
        init_cfg=dict(
            type='Pretrained',
            # 예시: COCO 데이터셋으로 학습된 MMDetection의 RetinaNet 모델 체크포인트
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth')
        ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    cotr=dict(
        type='COTR',
        num_kp=200,
        # --- 기존 cotr_args의 내용을 여기에 추가 ---
        max_corrs=1000,
        dim_feedforward=1024,
        backbone='resnet50',
        hidden_dim=312,
        dilation=False,
        dropout=0.1,
        nheads=8,
        layer='layer3',
        enc_layers=6,
        dec_layers=6,
        position_embedding='lin_sine',
        load_weights_freeze=False,
        # 가중치 로딩은 mmdet3d의 표준 방식인 init_cfg를 사용합니다.
        # 가중치 파일이 있다면 아래와 같이 설정합니다.
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/weights/backbone_base_corr_rev5.0_corrected.pth' # 예시 경로
            # checkpoint=None # 가중치 로딩이 필요 없을 경우
        )
    ),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256)
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(type='CopyImageToKey', key='img', new_key='img_original'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,       # <-- 2D 박스 로드 옵션 추가
        with_label=True,      # <-- 2D 라벨 로드 옵션 추가
        with_attr_label=False),
    dict(
        type='CustomImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointToMultiViewDepth', grid_config=grid_config, downsample=1),
    dict(type='CustomPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='CustomPack3DDetInputs',
        # class_names를 전달하여 라벨 변환을 활성화
        class_names=class_names,
        # 처리할 모든 키 목록
        keys=[
            'points', 'img', 'img_original','points_original',
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes','gt_labels',
        ],
        # 메타 정보로 처리할 키 목록
        meta_keys=[
            # --- ✨ 필수 메타 키 추가 ✨ ---
            'img_shape', 'ori_shape', 'pad_shape', 'scale_factor',
            # 기존 메타 키들
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats',
            'gt_KT','mis_RT','mis_KT','lidar_depth_gt','lidar_depth_mis','matched_uvset','perturbed_points',
            'ann_info_2d_per_cam' 
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d',
              'img_original','points_original','gt_KT','mis_RT','mis_KT',
            'lidar_depth_gt','lidar_depth_mis','matched_uvset','perturbed_points',],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality),),
    collate_fn=dict(type='custom_collate')# <-- 이 라인을 추가!
)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality),
    collate_fn=dict(type='custom_collate')# <-- 이 라인을 추가!)
)
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1,)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
del _base_.custom_hooks
