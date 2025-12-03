# import mmdet
# # from mmdet.models.dense_heads.retina_head

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
             'projects.BEVFusion.bevfusion.cotr',
             'projects.BEVFusion.bevfusion.my_hooks'],
    allow_failed_imports=False)

data_root = '/workspace/mmdetection3d/data/nuscenes/'

model = dict(
    type='BEVFusion',
    enable_selective_freezing=True,
    class_names=class_names,
    train_cfg=dict(
        complement_2d_gt=0.35, # <-- 커스텀 설정을 여기로 이동
        # 레퍼런스 코드에서 사용했던 다른 설정도 여기에 추가
        detection_proposal=dict(min_bbox_size=0),
        # # --- ✨ [핵심 수정] corr 모듈을 학습에서 제외(freeze)하도록 설정 ---
        # frozen_modules=['corr']
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
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=  # noqa: E251
        #     'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/work_dirs/extracted_backbones6/img_backbone_pretrained.pth'),  # noqa: E501)
        ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False),
        init_cfg=dict(type='Pretrained', checkpoint='data/work_dirs/extracted_backbones6/img_neck_pretrained.pth'),
        ),
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
            checkpoint='data/work_dirs/extracted_backbones6/img_bbox_head_pretrained.pth'),
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
    corr=dict(
        type='COTR',
        # frozen=True,  # <--- ✨✨ 여기에 frozen 플래그를 직접 추가합니다.
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
            # checkpoint='data/weights/backbone_base_corr_rev5.0_corrected.pth' # 예시 경로
            checkpoint='data/work_dirs/extracted_backbones6/corr_pretrained.pth' # 예시 경로
            # checkpoint=None # 가중치 로딩이 필요 없을 경우
        )
    ),
    z_estimator=dict(
        type='ZEstimator',
        enc_channels=312,
        uv_dim=2,
        hidden_dim=512,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/work_dirs/extracted_backbones6/z_estimator_pretrained.pth')
    ),
    calib_head=dict(
        type='CalibrationCorrectionHead',
        in_channels=312,
        # hidden_dim=256,
        # out_dim=6,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/work_dirs/extracted_backbones6/calib_head_pretrained.pth')
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
        with_attr_label=False,),
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
    # --- ✨ 여기에 새로운 Transform 추가! ✨ ---
    dict(type='GenerateUpdated2DAnnotations',
         classes=class_names, 
         visualize=False,
         vis_dir='vis_2d_detection_outputs'),
    dict(
        type='CustomPack3DDetInputs',
        # class_names를 전달하여 라벨 변환을 활성화
        class_names=class_names,
        # 처리할 모든 키 목록
        keys=[
            'points', 'img', 'img_original','points_original','perturbed_points',
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
            'gt_KT','mis_RT','mis_KT','lidar_depth_gt','lidar_depth_mis','matched_uvset',
            'ann_info_2d_per_cam' ,'ann_info_aug_2d_per_cam','img_aug_params',
            'camera2lidar', 'broken_camera2lidar', 'broken_camera_intrinsics',
            'gt_delta_rot','gt_delta_trans',
        ])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    # <<< [추가] 원본 이미지를 보존하기 위해 복사
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
    # <<< [추가] 평가를 위해 2D/3D 어노테이션 로드
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_attr_label=False,),
    # <<< [수정] Augmentation 타입을 CustomImageAug3D로 통일 (랜덤 옵션은 비활성화)
    dict(
        type='CustomImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48], # 테스트 시에는 고정된 크기 사용
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    # <<< [추가] 학습 파이프라인과 동일한 단계 추가
    dict(type='PointToMultiViewDepth', grid_config=grid_config, downsample=1),
    # <<< [수정] 필터 타입을 CustomPointsRangeFilter로 통일 (또는 유지)
    dict(type='CustomPointsRangeFilter', point_cloud_range=point_cloud_range),
    # # <<< [추가] 학습 시와 동일한 필터링 적용
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(
    #     type='ObjectNameFilter',
    #     classes=[
    #         'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    #         'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    #     ]),
    # <<< [핵심 수정] Pack 단계를 CustomPack3DDetInputs로 통일하고 모든 키 포함
    dict(
        type='CustomPack3DDetInputs',
        class_names=class_names,
        keys=[
            'points', 'img', 'img_original', 'points_original','perturbed_points',
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels',
        ],
        meta_keys=[
            # --- ✨ 필수 메타 키 추가 ✨ ---
            'img_shape', 'ori_shape', 'pad_shape', 'scale_factor',
            # 기존 메타 키들
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats',
            'gt_KT','mis_RT','mis_KT','lidar_depth_gt','lidar_depth_mis','matched_uvset',
            'ann_info_2d_per_cam' ,'ann_info_aug_2d_per_cam','img_aug_params',
            'camera2lidar', 'broken_camera2lidar', 'broken_camera_intrinsics',
            'gt_delta_rot','gt_delta_trans',
        ])
]

# 1. 각 Dataloader를 모든 정보(ann_file 포함)와 함께 명확하게 정의
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            type='NuScenesDataset',
            data_root=data_root,
            # ann_file='nuscenes_infos_train_new_with_2d.pkl', # 경로 명시
            ann_file='debug_infos_train_with_2d.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    # # ✨✨ 핵심 수정: Sampler를 DefaultSampler로 지정하고 shuffle=False 설정
    # sampler=dict(
    #     type='DefaultSampler',
    #     shuffle=False
    # ),
    collate_fn=dict(type='custom_collate')
)

val_dataloader = dict(
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        # ann_file='nuscenes_infos_val_new_with_2d.pkl', # 경로 명시
        ann_file='debug_infos_val_with_2d.pkl',
        pipeline=val_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=False,),
    collate_fn=dict(type='custom_collate')
)

# 2. val 설정을 test에서도 사용하도록 명시적으로 재지정 (가장 중요!)
test_dataloader = val_dataloader

# 3. Evaluator도 명시적으로 재지정
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    # ann_file=data_root + 'nuscenes_infos_val_new_with_2d.pkl',
    ann_file=data_root + 'debug_infos_val_with_2d.pkl',
    metric='bbox',
    version='v1.0-trainval',
    collect_dir='test_results_tmp',)  # <-- 이 라인을 추가하세요.)

# test_evaluator = val_evaluator

# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.33333333,
#         by_epoch=False,
#         begin=0,
#         end=500),
#     dict(
#         type='CosineAnnealingLR',
#         begin=0,
#         T_max=6,
#         end=6,
#         by_epoch=True,
#         eta_min_ratio=1e-4,
#         convert_to_iter_based=True),
#     # momentum scheduler
#     # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
#     # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
#     dict(
#         type='CosineAnnealingMomentum',
#         eta_min=0.85 / 0.95,
#         begin=0,
#         end=2.4,
#         by_epoch=True,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingMomentum',
#         eta_min=1,
#         begin=2.4,
#         end=6,
#         by_epoch=True,
#         convert_to_iter_based=True)
# ]

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
        T_max=20,
        end=20,
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
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=24,)
val_cfg = dict()
test_cfg = dict()

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
#     clip_grad=dict(max_norm=35, norm_type=2))

# --- ✨ 핵심 수정: 옵티마이저 설정을 변경하여 모듈별로 다른 학습률 적용 ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01), # 1.9033e-04 # 초기 : 0.0002 1.6693e-04 1.6541e-04 1.6524e-04
    # paramwise_cfg를 통해 특정 파라미터 그룹에 다른 학습률을 설정합니다.
    paramwise_cfg=dict(
        custom_keys={
            # 이미지 백본은 사전 학습된 가중치를 사용하므로, 더 작은 학습률로 미세 조정합니다.
            'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
            # 'z_estimator': dict(lr_mult=0.1), # z_estimator의 학습률만 10배로
            'img_neck'  :dict(lr_mult=0.1, decay_mult=1.0),
            'img_bbox_head': dict(lr_mult=0.1, decay_mult=1.0),
            # corr 모듈은 사전 학습된 가중치를 사용하므로, 더 작은 학습률로 미세 조정합니다.
            # 'corr': dict(lr_mult=0.1, decay_mult=1.0),
            # # pts_backbone
            'pts_voxel_layer' : dict(lr_mult=0.1, decay_mult=1.0),
            'pts_voxel_encoder' :dict(lr_mult=0.1, decay_mult=1.0),
            'pts_middle_encoder' : dict(lr_mult=0.1, decay_mult=1.0),
            'pts_backbone' : dict(lr_mult=0.1, decay_mult=1.0),
            'pts_neck' : dict(lr_mult=0.1, decay_mult=1.0),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook',
                interval=50,
                ),
    # checkpoint=dict(type='CheckpointHook', interval=1),
    # checkpoint=dict(
    #     type='CheckpointHook',
    #     interval=500,      # 1000번의 이터레이션마다 저장
    #     by_epoch=False,
    #     max_keep_ckpts=3,),    # 👈 이 부분을 False로 변경하는 것이 핵심입니다.
    checkpoint=dict(
        type='CheckpointHook',
        interval=3000,           # 500 이터레이션마다 체크포인트 저장 조건 확인
        by_epoch=False,
        save_best='train/loss',   # 'val/loss'를 기준으로 가장 좋은 모델을 저장
        rule='less',            # loss는 낮을수록 좋으므로 'less'로 설정
        max_keep_ckpts=3,       # 가장 좋은 체크포인트 3개만 유지
    ),
    # --- ▼▼▼ 이 부분을 아래와 같이 수정하세요 ▼▼▼ ---
    visualization=dict(
        type='Det3DVisualizationHook',
        draw=True,      # <-- BEV 시각화 활성화 스위치
        interval=1,      # <-- 매 1개 샘플마다 시각화 결과 저장
        test_out_dir='visualization_results' , # <-- 이 라인을 추가!
    ))
del _base_.custom_hooks

# load_from =  "data/weights/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
load_from =  "data/work_dirs/bevfusion/20251127_renew_10deg_0.75m_corr_pretrained_v5.0/iter_159000.pth"
# load_from = None
resume_from = None

# log_level = 'WARNING' 
# 1. vis_backends 리스트를 먼저 정의합니다.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

# 2. visualizer 딕셔너리에 'name' 필드를 추가하고, 위에서 정의한 백엔드를 연결합니다.
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'  # <-- 이 라인이 추가되었습니다!
)

# custom_hooks = [
#     dict(type='ValidateBeforeTrainHook')
# ]