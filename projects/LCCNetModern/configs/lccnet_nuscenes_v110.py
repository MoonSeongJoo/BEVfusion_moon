# ============================================================
# projects/LCCNetModern/configs/lccnet_nuscenes_v110.py
#
# LCCNet standalone calibration training / validation config
#
# Base geometry:
#   Official BEVFusion v1.1.0
#
# Purpose:
#   O-5C
#   - nuScenes 6-camera RGB
#   - LiDAR + 9 sweeps
#   - GT camera-to-LiDAR extrinsic
#   - image augmentation matrix
#   - LCCNet broken-depth generation
#
# IMPORTANT:
#   This config does NOT train BEVFusion.
#
#   BEVFusion detector backbone / neck / TransFusionHead /
#   detection labels are intentionally removed.
#
# ============================================================


# ============================================================
# Base runtime
# ============================================================

_base_ = [
    'mmdet3d::_base_/default_runtime.py'
]


# ============================================================
# Custom imports
#
# Required for:
#   BEVLoadMultiViewImageFromFiles
#   ImageAug3D
#
# These are from Official BEVFusion v1.1.0 port.
# ============================================================

custom_imports = dict(
    imports=[
        'projects.OfficialBEVFusion.bevfusion',
    ],
    allow_failed_imports=False,
)


default_scope = 'mmdet3d'


# ============================================================
# Dataset
# ============================================================

dataset_type = 'NuScenesDataset'

data_root = 'data/nuscenes/'

backend_args = None


# ============================================================
# nuScenes class names
#
# LCCNet itself does NOT use object labels.
# We keep this metadata only because NuScenesDataset expects
# standard dataset metadata.
# ============================================================

class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

metainfo = dict(
    classes=class_names,
)


# ============================================================
# Sensor modality
# ============================================================

input_modality = dict(
    use_lidar=True,
    use_camera=True,
)


# ============================================================
# Dataset paths
#
# Same 6-camera + LIDAR_TOP structure used by Official
# BEVFusion.
# ============================================================

data_prefix = dict(

    pts='samples/LIDAR_TOP',

    CAM_FRONT='samples/CAM_FRONT',

    CAM_FRONT_LEFT=(
        'samples/CAM_FRONT_LEFT'
    ),

    CAM_FRONT_RIGHT=(
        'samples/CAM_FRONT_RIGHT'
    ),

    CAM_BACK='samples/CAM_BACK',

    CAM_BACK_RIGHT=(
        'samples/CAM_BACK_RIGHT'
    ),

    CAM_BACK_LEFT=(
        'samples/CAM_BACK_LEFT'
    ),

    sweeps='sweeps/LIDAR_TOP',
)


# ============================================================
# Point cloud range
#
# Keep identical to Official BEVFusion.
# ============================================================

point_cloud_range = [
    -54.0,
    -54.0,
    -5.0,
    54.0,
    54.0,
    3.0,
]


# ============================================================
# Input image size
#
# Keep identical to O-0 ~ O-5A benchmark.
# ============================================================

image_size = [
    256,
    704,
]


# ============================================================
# Minimal "model" section
#
# train_lccnet_nuscenes.py only accesses:
#
#     cfg.model.data_preprocessor
#
# Therefore BEVFusion itself is NOT built.
#
# IMPORTANT:
# No voxelize_cfg here.
#
# LCCNet training only needs:
#   normalized image tensor
#   raw point tensor
#
# Building BEVFusion voxel features would waste GPU memory
# and computation.
# ============================================================

model = dict(

    data_preprocessor=dict(

        type='Det3DDataPreprocessor',

        # Same normalization as Official BEVFusion
        mean=[
            123.675,
            116.28,
            103.53,
        ],

        std=[
            58.395,
            57.12,
            57.375,
        ],

        # IMPORTANT:
        # Official BEVFusion's custom image loader loads RGB.
        # Therefore we do not perform BGR -> RGB conversion.
        bgr_to_rgb=False,

        pad_size_divisor=32,
    )
)


# ============================================================
# Common meta keys required by O-5C
#
# Most important:
#
#   cam2img
#   lidar2cam
#   lidar2img
#   cam2lidar
#   img_aug_matrix
#   lidar_aug_matrix
#
# train_lccnet_nuscenes.py uses these to generate:
#
#   T_broken
#   broken lidar2img
#   broken sparse depth
#
# lidar_aug_matrix may not exist when no LiDAR augmentation
# is applied. train_lccnet_nuscenes.py already substitutes
# identity in that case.
# ============================================================

lccnet_meta_keys = [

    'cam2img',

    'ori_cam2img',

    'lidar2cam',

    'lidar2img',

    'cam2lidar',

    'img_aug_matrix',

    'lidar_aug_matrix',

    'box_type_3d',

    'sample_idx',

    'lidar_path',

    'img_path',
]


# ============================================================
# TRAIN PIPELINE
#
# Differences from Official BEVFusion detector training:
#
# Removed:
#
#   LoadAnnotations3D
#   ObjectSampling
#   ObjectRangeFilter
#   ObjectNameFilter
#   GridMask
#   GT bbox / labels
#
# Also intentionally removed:
#
#   GlobalRotScaleTrans
#   RandomFlip3D
#
# Reason:
#
# The calibration training perturbation is independently
# generated in calib_geometry.py:
#
#     T_broken = Delta_GT @ T_GT
#
# Additional global LiDAR augmentation is unnecessary for
# O-5C and introduces another geometry variable.
#
# ImageAug3D is retained because RGB/depth paired image
# augmentation is useful and img_aug_matrix makes the
# projection geometrically consistent.
# ============================================================

train_pipeline = [

    # --------------------------------------------------------
    # 1. Six nuScenes camera images
    # --------------------------------------------------------

    dict(
        type='BEVLoadMultiViewImageFromFiles',

        to_float32=True,

        color_type='color',

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 2. Current LIDAR_TOP keyframe
    #
    # IMPORTANT:
    #
    # No:
    #   reduce_beams
    #   load_augmented
    #
    # because current MMDetection3D API does not accept them.
    # --------------------------------------------------------

    dict(
        type='LoadPointsFromFile',

        coord_type='LIDAR',

        load_dim=5,

        use_dim=5,

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 3. 9 previous sweeps
    #
    # Keep same multi-sweep condition as Official BEVFusion
    # and O-5A broken-depth generation.
    # --------------------------------------------------------

    dict(
        type='LoadPointsFromMultiSweeps',

        sweeps_num=9,

        load_dim=5,

        use_dim=5,

        pad_empty_sweeps=True,

        remove_close=True,

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 4. Image augmentation
    #
    # Same image size and augmentation range as the original
    # Official BEVFusion TRAIN pipeline.
    #
    # img_aug_matrix is generated by this transform.
    # --------------------------------------------------------

    dict(
        type='ImageAug3D',

        final_dim=[
            256,
            704,
        ],

        resize_lim=[
            0.38,
            0.55,
        ],

        bot_pct_lim=[
            0.0,
            0.0,
        ],

        rot_lim=[
            -5.4,
            5.4,
        ],

        rand_flip=True,

        is_train=True,
    ),


    # --------------------------------------------------------
    # 5. Keep same LiDAR spatial range as BEVFusion
    # --------------------------------------------------------

    dict(
        type='PointsRangeFilter',

        point_cloud_range=(
            point_cloud_range
        ),
    ),


    # --------------------------------------------------------
    # 6. Pack only inputs required for calibration
    #
    # No detection GT is needed.
    # --------------------------------------------------------

    dict(
        type='Pack3DDetInputs',

        keys=[
            'points',
            'img',
        ],

        meta_keys=(
            lccnet_meta_keys
        ),
    ),
]


# ============================================================
# VALIDATION PIPELINE
#
# Deterministic image processing.
#
# The calibration perturbation itself is NOT generated here.
#
# O-5C-8 will use:
#
#   max_rot_deg = 10
#   max_trans_m = 0.75
#   seed = 20260811
#
# through our deterministic O-3 perturbation path.
#
# This pipeline therefore only guarantees a fixed sensor
# input preprocessing condition.
# ============================================================

val_pipeline = [

    # --------------------------------------------------------
    # 1. Camera images
    # --------------------------------------------------------

    dict(
        type='BEVLoadMultiViewImageFromFiles',

        to_float32=True,

        color_type='color',

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 2. Current LiDAR
    # --------------------------------------------------------

    dict(
        type='LoadPointsFromFile',

        coord_type='LIDAR',

        load_dim=5,

        use_dim=5,

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 3. 9 sweeps
    # --------------------------------------------------------

    dict(
        type='LoadPointsFromMultiSweeps',

        sweeps_num=9,

        load_dim=5,

        use_dim=5,

        pad_empty_sweeps=True,

        remove_close=True,

        test_mode=True,

        backend_args=backend_args,
    ),


    # --------------------------------------------------------
    # 4. Deterministic Official-style image augmentation
    # --------------------------------------------------------

    dict(
        type='ImageAug3D',

        final_dim=[
            256,
            704,
        ],

        resize_lim=[
            0.48,
            0.48,
        ],

        bot_pct_lim=[
            0.0,
            0.0,
        ],

        rot_lim=[
            0.0,
            0.0,
        ],

        rand_flip=False,

        is_train=False,
    ),


    # --------------------------------------------------------
    # 5. Same point cloud range
    # --------------------------------------------------------

    dict(
        type='PointsRangeFilter',

        point_cloud_range=(
            point_cloud_range
        ),
    ),


    # --------------------------------------------------------
    # 6. Pack calibration inputs only
    # --------------------------------------------------------

    dict(
        type='Pack3DDetInputs',

        keys=[
            'points',
            'img',
        ],

        meta_keys=(
            lccnet_meta_keys
        ),
    ),
]


# ============================================================
# TRAIN DATALOADER
#
# IMPORTANT:
#
# Use v1.1.0-converted nuScenes train info.
#
# Expected:
#
#   data/nuscenes/
#       v110_infos/
#           nuscenes_infos_train.pkl
#
# train_lccnet_nuscenes.py currently overrides batch_size
# and num_workers from CLI, but safe defaults are kept here.
# ============================================================

train_dataloader = dict(

    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',

        shuffle=True,
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file=(
            'v110_infos/'
            'nuscenes_infos_train.pkl'
        ),

        pipeline=train_pipeline,

        metainfo=metainfo,

        modality=input_modality,

        filter_empty_gt=False,

        test_mode=False,

        data_prefix=data_prefix,

        box_type_3d='LiDAR',

        backend_args=backend_args,
    ),
)


# ============================================================
# VALIDATION DATALOADER
#
# Uses the same v1.1.0 validation info already used for the
# Official BEVFusion reproduction benchmark.
# ============================================================

val_dataloader = dict(

    batch_size=1,

    num_workers=0,

    persistent_workers=False,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',

        shuffle=False,
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file=(
            'v110_infos/'
            'nuscenes_infos_val.pkl'
        ),

        pipeline=val_pipeline,

        metainfo=metainfo,

        modality=input_modality,

        test_mode=True,

        data_prefix=data_prefix,

        box_type_3d='LiDAR',

        backend_args=backend_args,
    ),
)


# ============================================================
# Convenience alias
# ============================================================

test_dataloader = val_dataloader


# ============================================================
# LCCNet experiment metadata
#
# train_lccnet_nuscenes.py currently controls these values
# through CLI / source code.
#
# This block is mainly included to make the experiment
# configuration self-documenting.
# ============================================================

lccnet_settings = dict(

    image_size=[
        256,
        704,
    ],

    num_cameras=6,

    depth_scale=50.0,

    max_rot_deg=10.0,

    max_trans_m=0.75,

    seed=20260811,

    # Broken calibration convention:
    #
    #     T_broken = Delta_GT @ T_GT
    #
    broken_transform_convention=(
        'left_multiply'
    ),

    # LCCNet target:
    #
    #     C_GT = inv(Delta_GT)
    #     target = se3.log(C_GT)
    #
    target_convention=(
        'inverse_delta_se3_log'
    ),
)


# ============================================================
# No NuScenes detection evaluator is defined here.
#
# O-5C validation measures:
#
#   rotation calibration residual [deg]
#   translation calibration residual [m]
#   L1 calibration loss
#
# not detector mAP / NDS.
#
# Detection evaluation belongs to O-5D:
#
#   LCCNet
#      ↓
#   corrected C2L
#      ↓
#   frozen Official BEVFusion
#      ↓
#   mAP / NDS
# ============================================================


# ============================================================
# End of config
# ============================================================