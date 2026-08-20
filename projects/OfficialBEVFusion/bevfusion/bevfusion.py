from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization

import hashlib
import math


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        vtransform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        calibration_mode='official',
        perturb_max_rot_deg=10.0,
        perturb_max_trans_m=0.75,
        perturb_seed=20260811,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.calibration_mode = calibration_mode
        self.perturb_max_rot_deg = perturb_max_rot_deg
        self.perturb_max_trans_m = perturb_max_trans_m
        self.perturb_seed = perturb_seed

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.vtransform = MODELS.build(vtransform)
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(fusion_layer)

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)
        # hard code here where using converted checkpoint of original
        # implementation of `BEVFusion`
        self.use_converted_checkpoint = True

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.vtransform(
            x,
            points,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes
    
    def _sample_calibration_delta(
        self,
        sample_key,
        cam_idx,
        like_tensor,
    ):
        """Generate deterministic SE(3) perturbation.

        Rotation:
            rx, ry, rz ~ U[-max_rot, +max_rot] degree

        Translation:
            tx, ty, tz ~ U[-max_trans, +max_trans] meter

        The seed depends on:
            global perturb_seed
            sample identity
            camera index

        Therefore the same sample/camera always gets
        exactly the same perturbation even with DDP.
        """

        key = (
            f"{self.perturb_seed}:"
            f"{sample_key}:"
            f"{cam_idx}"
        )

        digest = hashlib.sha1(
            key.encode('utf-8')
        ).digest()

        seed = int.from_bytes(
            digest[:8],
            byteorder='little',
            signed=False,
        )

        seed = seed % (2**63 - 1)

        generator = torch.Generator(
            device='cpu'
        )

        generator.manual_seed(seed)

        rnd = torch.rand(
            6,
            generator=generator,
            dtype=torch.float64,
        )

        # ------------------------------------------------------
        # Rotation [deg]
        # ------------------------------------------------------

        rot_deg = (
            2.0 * rnd[:3] - 1.0
        ) * self.perturb_max_rot_deg

        # ------------------------------------------------------
        # Translation [m]
        # ------------------------------------------------------

        trans_m = (
            2.0 * rnd[3:] - 1.0
        ) * self.perturb_max_trans_m

        rx_deg = float(rot_deg[0])
        ry_deg = float(rot_deg[1])
        rz_deg = float(rot_deg[2])

        tx = float(trans_m[0])
        ty = float(trans_m[1])
        tz = float(trans_m[2])

        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        # ------------------------------------------------------
        # Euler convention:
        #
        # R = Rz @ Ry @ Rx
        # ------------------------------------------------------

        Rx = like_tensor.new_tensor([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx,  cx],
        ])

        Ry = like_tensor.new_tensor([
            [ cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ])

        Rz = like_tensor.new_tensor([
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ])

        R = Rz @ Ry @ Rx

        delta = torch.eye(
            4,
            device=like_tensor.device,
            dtype=like_tensor.dtype,
        )

        delta[:3, :3] = R

        delta[:3, 3] = like_tensor.new_tensor([
            tx,
            ty,
            tz,
        ])

        params = {
            'rx_deg': rx_deg,
            'ry_deg': ry_deg,
            'rz_deg': rz_deg,
            'tx_m': tx,
            'ty_m': ty,
            'tz_m': tz,
        }

        return delta, params
    

    def _build_broken_camera2lidar(
        self,
        camera2lidar_gt,
        batch_input_metas,
    ):
        """Create deterministic broken Camera->LiDAR extrinsics.

        Convention:

            T_broken = DeltaT_GT @ T_GT
        """

        B, N = camera2lidar_gt.shape[:2]

        broken_camera2lidar = torch.empty_like(
            camera2lidar_gt
        )

        delta_gt = torch.empty_like(
            camera2lidar_gt
        )

        first_debug_params = None

        for b in range(B):

            meta = batch_input_metas[b]

            sample_key = meta.get(
                'token',
                meta.get(
                    'sample_idx',
                    f'batch_{b}'
                )
            )

            sample_key = str(sample_key)

            for cam_idx in range(N):

                delta, params = (
                    self._sample_calibration_delta(
                        sample_key,
                        cam_idx,
                        camera2lidar_gt,
                    )
                )

                # ==================================================
                # IMPORTANT benchmark convention
                #
                # T_broken = DeltaT_GT @ T_GT
                # ==================================================

                broken_camera2lidar[
                    b, cam_idx
                ] = (
                    delta
                    @ camera2lidar_gt[b, cam_idx]
                )

                delta_gt[
                    b, cam_idx
                ] = delta

                if (
                    first_debug_params is None
                    and b == 0
                    and cam_idx == 0
                ):
                    first_debug_params = params

        return (
            broken_camera2lidar,
            delta_gt,
            first_debug_params,
        )

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        external_calib = kwargs.get(
            'external_calib',
            None
        )

        feats = self.extract_feat(
            batch_inputs_dict,
            batch_input_metas,
            external_calib=external_calib,
        )

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)
            if self.use_converted_checkpoint:
                outputs[0]['bboxes_3d'].tensor[:, 6] = -outputs[0][
                    'bboxes_3d'].tensor[:, 6] - np.pi / 2
                outputs[0]['bboxes_3d'].tensor[:, 3:5] = outputs[0][
                    'bboxes_3d'].tensor[:, [4, 3]]

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    # def extract_feat(
    #     self,
    #     batch_inputs_dict,
    #     batch_input_metas,
    #     **kwargs,
    # ):
    #     imgs = batch_inputs_dict.get('imgs', None)
    #     points = batch_inputs_dict.get('points', None)

    #     lidar2image, camera_intrinsics, camera2lidar = [], [], []
    #     img_aug_matrix, lidar_aug_matrix = [], []
    #     for i, meta in enumerate(batch_input_metas):
    #         lidar2image.append(meta['lidar2img'])
    #         camera_intrinsics.append(meta['cam2img'])
    #         camera2lidar.append(meta['cam2lidar'])
    #         img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
    #         lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

    #     lidar2image = imgs.new_tensor(np.asarray(lidar2image))
    #     camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
    #     camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
    #     img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
    #     lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
    #     img_feature = self.extract_img_feat(imgs, points, lidar2image,
    #                                         camera_intrinsics, camera2lidar,
    #                                         img_aug_matrix, lidar_aug_matrix,
    #                                         batch_input_metas)
    #     pts_feature = self.extract_pts_feat(batch_inputs_dict)

    #     features = [img_feature, pts_feature]

    #     if self.fusion_layer is not None:
    #         x = self.fusion_layer(features)
    #     else:
    #         assert len(features) == 1, features
    #         x = features[0]

    #     x = self.pts_backbone(x)
    #     x = self.pts_neck(x)

    #     return x

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)

        lidar2image_meta = []
        camera_intrinsics = []
        camera2lidar = []

        img_aug_matrix = []
        lidar_aug_matrix = []

        for i, meta in enumerate(batch_input_metas):

            # ------------------------------------------------------
            # Original GT calibration from metadata
            # ------------------------------------------------------
            lidar2image_meta.append(
                meta['lidar2img']
            )

            camera_intrinsics.append(
                meta['cam2img']
            )

            camera2lidar.append(
                meta['cam2lidar']
            )

            # ------------------------------------------------------
            # Keep official augmentation matrices unchanged
            # ------------------------------------------------------
            img_aug_matrix.append(
                meta.get(
                    'img_aug_matrix',
                    np.eye(4)
                )
            )

            lidar_aug_matrix.append(
                meta.get(
                    'lidar_aug_matrix',
                    np.eye(4)
                )
            )


        # ==========================================================
        # Convert metadata into tensors
        # ==========================================================

        lidar2image_meta = imgs.new_tensor(
            np.asarray(lidar2image_meta)
        )

        camera_intrinsics = imgs.new_tensor(
            np.asarray(camera_intrinsics)
        )

        camera2lidar = imgs.new_tensor(
            np.asarray(camera2lidar)
        )

        img_aug_matrix = imgs.new_tensor(
            np.asarray(img_aug_matrix)
        )

        lidar_aug_matrix = imgs.new_tensor(
            np.asarray(lidar_aug_matrix)
        )

        # ==========================================================
        # DEFAULT = Official O-0
        # ==========================================================

        lidar2image = lidar2image_meta

        # ==========================================================
        # Calibration routing
        # ==========================================================

        if self.calibration_mode == 'gt_reinject':

            # ------------------------------------------------------
            # Gate O-2
            #
            # DO NOT use metadata lidar2img.
            #
            # Reconstruct lidar2img only from:
            #     GT cam2lidar
            #     GT cam2img
            # ------------------------------------------------------

            calib = self._build_calib_dict_from_cam2lidar(
                camera2lidar,
                camera_intrinsics,
            )

            lidar2image = calib['lidar2img']
            camera_intrinsics = calib['cam2img']
            camera2lidar = calib['cam2lidar']
        
        elif self.calibration_mode == 'broken':

            # ======================================================
            # Gate O-3 Broken
            # ======================================================

            camera2lidar_gt = camera2lidar

            (
                broken_camera2lidar,
                delta_gt,
                first_debug_params,
            ) = self._build_broken_camera2lidar(
                camera2lidar_gt,
                batch_input_metas,
            )

            # ------------------------------------------------------
            # CRITICAL:
            #
            # ALL broken geometry is derived from exactly
            # the same broken Camera->LiDAR extrinsic.
            # ------------------------------------------------------

            calib = self._build_calib_dict_from_cam2lidar(
                broken_camera2lidar,
                camera_intrinsics,
            )

            lidar2image = calib['lidar2img']
            camera_intrinsics = calib['cam2img']
            camera2lidar = calib['cam2lidar']

            if not hasattr(
                self,
                '_gate_o3_checked'
            ):

                # ------------------------------------------------------
                # Check 1:
                # T_broken must actually differ from T_GT.
                # ------------------------------------------------------

                err_broken_vs_clean = (
                    broken_camera2lidar
                    - camera2lidar_gt
                ).abs().max().item()


                # ------------------------------------------------------
                # Check 2:
                #
                # Since:
                #
                # T_broken = Delta @ T_GT
                #
                # Delta_recovered =
                #     T_broken @ inv(T_GT)
                #
                # must equal Delta.
                # ------------------------------------------------------

                delta_recovered = (
                    broken_camera2lidar
                    @ torch.linalg.inv(
                        camera2lidar_gt
                    )
                )

                err_delta = (
                    delta_recovered
                    - delta_gt
                ).abs().max().item()


                # ------------------------------------------------------
                # Check 3:
                #
                # Oracle algebra preview:
                #
                # inv(Delta) @ T_broken
                # must restore T_GT.
                # ------------------------------------------------------

                oracle_camera2lidar = (
                    torch.linalg.inv(delta_gt)
                    @ broken_camera2lidar
                )

                err_oracle = (
                    oracle_camera2lidar
                    - camera2lidar_gt
                ).abs().max().item()


                # ------------------------------------------------------
                # Check 4:
                # Broken projection must differ from Clean projection.
                # ------------------------------------------------------

                err_lidar2img_change = (
                    lidar2image
                    - lidar2image_meta
                ).abs().max().item()


                print(
                    "\n[GATE O-3]"
                    "\nBroken Calibration"
                )

                print(
                    "[GATE O-3] "
                    "max_rot_deg =",
                    self.perturb_max_rot_deg
                )

                print(
                    "[GATE O-3] "
                    "max_trans_m =",
                    self.perturb_max_trans_m
                )

                print(
                    "[GATE O-3] "
                    "seed =",
                    self.perturb_seed
                )

                print(
                    "[GATE O-3] "
                    "first camera perturb =",
                    first_debug_params
                )

                print(
                    "[GATE O-3] "
                    "Broken C2L vs GT C2L = "
                    f"{err_broken_vs_clean:.8e}"
                )

                print(
                    "[GATE O-3] "
                    "Recovered Delta vs GT Delta = "
                    f"{err_delta:.8e}"
                )

                print(
                    "[GATE O-3] "
                    "Oracle restore GT error = "
                    f"{err_oracle:.8e}"
                )

                print(
                    "[GATE O-3] "
                    "Broken lidar2img vs Clean lidar2img = "
                    f"{err_lidar2img_change:.8e}"
                )


                # ------------------------------------------------------
                # Geometry safety gates
                # ------------------------------------------------------

                assert err_delta < 1e-5, (
                    "Gate O-3 FAILED: "
                    "Delta reconstruction mismatch"
                )

                assert err_oracle < 1e-5, (
                    "Gate O-3 FAILED: "
                    "Oracle algebra mismatch"
                )

                assert err_lidar2img_change > 1e-3, (
                    "Gate O-3 FAILED: "
                    "Broken projection did not change"
                )

                self._gate_o3_checked = True
        
        elif self.calibration_mode == 'oracle':

            # ======================================================
            # Gate O-4 Oracle
            # ======================================================

            camera2lidar_gt = camera2lidar

            # ------------------------------------------------------
            # Generate EXACTLY the same deterministic perturbation
            # used in Gate O-3.
            # ------------------------------------------------------

            (
                broken_camera2lidar,
                delta_gt,
                first_debug_params,
            ) = self._build_broken_camera2lidar(
                camera2lidar_gt,
                batch_input_metas,
            )

            # ------------------------------------------------------
            # Perfect correction:
            #
            # T_broken = Delta_GT @ T_GT
            #
            # therefore:
            #
            # T_oracle
            # = inv(Delta_GT) @ T_broken
            # = T_GT
            # ------------------------------------------------------

            oracle_camera2lidar = (
                torch.linalg.inv(delta_gt)
                @ broken_camera2lidar
            )

            # ------------------------------------------------------
            # Derive ALL downstream calibration from oracle C2L.
            # ------------------------------------------------------

            calib = self._build_calib_dict_from_cam2lidar(
                oracle_camera2lidar,
                camera_intrinsics,
            )

            lidar2image = calib['lidar2img']
            camera_intrinsics = calib['cam2img']
            camera2lidar = calib['cam2lidar']
        
            if not hasattr(
                self,
                '_gate_o4_checked'
            ):

                # ------------------------------------------------------
                # Check 1:
                # Oracle extrinsic must recover GT extrinsic.
                # ------------------------------------------------------

                err_oracle_c2l = (
                    oracle_camera2lidar
                    - camera2lidar_gt
                ).abs().max().item()


                # ------------------------------------------------------
                # Check 2:
                # Oracle projection must recover clean projection.
                # ------------------------------------------------------

                err_oracle_lidar2img = (
                    lidar2image
                    - lidar2image_meta
                ).abs().max().item()


                # ------------------------------------------------------
                # Check 3:
                # Broken itself must actually be non-zero.
                # This prevents accidentally testing identity perturbation.
                # ------------------------------------------------------

                err_broken_vs_gt = (
                    broken_camera2lidar
                    - camera2lidar_gt
                ).abs().max().item()


                print(
                    "\n[GATE O-4]"
                    "\nOracle Calibration Recovery"
                )

                print(
                    "[GATE O-4] "
                    "max_rot_deg =",
                    self.perturb_max_rot_deg
                )

                print(
                    "[GATE O-4] "
                    "max_trans_m =",
                    self.perturb_max_trans_m
                )

                print(
                    "[GATE O-4] "
                    "seed =",
                    self.perturb_seed
                )

                print(
                    "[GATE O-4] "
                    "first perturb =",
                    first_debug_params
                )

                print(
                    "[GATE O-4] "
                    "Broken C2L vs GT C2L = "
                    f"{err_broken_vs_gt:.8e}"
                )

                print(
                    "[GATE O-4] "
                    "Oracle C2L vs GT C2L = "
                    f"{err_oracle_c2l:.8e}"
                )

                print(
                    "[GATE O-4] "
                    "Oracle lidar2img vs Clean lidar2img = "
                    f"{err_oracle_lidar2img:.8e}"
                )


                assert err_broken_vs_gt > 1e-3, (
                    "Gate O-4 FAILED: "
                    "Broken perturbation is not active"
                )

                assert err_oracle_c2l < 1e-5, (
                    "Gate O-4 FAILED: "
                    "Oracle extrinsic did not recover GT"
                )

                assert err_oracle_lidar2img < 1e-3, (
                    "Gate O-4 FAILED: "
                    "Oracle projection did not recover Clean"
                )

                self._gate_o4_checked = True
        
        elif self.calibration_mode == 'lccnet_depth_debug':

            camera2lidar_gt = camera2lidar

            (
                broken_camera2lidar,
                delta_gt,
                first_debug_params,
            ) = self._build_broken_camera2lidar(
                camera2lidar_gt,
                batch_input_metas,
            )

            broken_calib = (
                self._build_calib_dict_from_cam2lidar(
                    broken_camera2lidar,
                    camera_intrinsics,
                )
            )

            clean_calib = (
                self._build_calib_dict_from_cam2lidar(
                    camera2lidar_gt,
                    camera_intrinsics,
                )
            )

            clean_depth = (
                self._project_lidar_to_sparse_depth(
                    points,
                    clean_calib['lidar2img'],
                    img_aug_matrix,
                    lidar_aug_matrix,
                    image_hw=(
                        self.vtransform.image_size[0],
                        self.vtransform.image_size[1],
                    ),
                )
            )

            broken_depth = (
                self._project_lidar_to_sparse_depth(
                    points,
                    broken_calib['lidar2img'],
                    img_aug_matrix,
                    lidar_aug_matrix,
                    image_hw=(
                        self.vtransform.image_size[0],
                        self.vtransform.image_size[1],
                    ),
                )
            )

            lidar2image = broken_calib['lidar2img']
            camera_intrinsics = broken_calib['cam2img']
            camera2lidar = broken_calib['cam2lidar']

            if not hasattr(
                self,
                '_gate_o5a_checked'
            ):

                clean_nonzero = (
                    clean_depth > 0
                ).sum().item()

                broken_nonzero = (
                    broken_depth > 0
                ).sum().item()

                diff = (
                    clean_depth
                    - broken_depth
                ).abs()

                diff_nonzero = (
                    diff > 1e-5
                ).sum().item()

                max_diff = diff.max().item()

                print(
                    "\n[GATE O-5A]"
                    "\nPhysically-consistent "
                    "misaligned depth"
                )

                print(
                    "[GATE O-5A] clean depth shape =",
                    tuple(clean_depth.shape)
                )

                print(
                    "[GATE O-5A] broken depth shape =",
                    tuple(broken_depth.shape)
                )

                print(
                    "[GATE O-5A] clean nonzero =",
                    clean_nonzero
                )

                print(
                    "[GATE O-5A] broken nonzero =",
                    broken_nonzero
                )

                print(
                    "[GATE O-5A] changed pixels =",
                    diff_nonzero
                )

                print(
                    "[GATE O-5A] max depth difference =",
                    max_diff
                )

                print(
                    "[GATE O-5A] finite clean =",
                    torch.isfinite(
                        clean_depth
                    ).all().item()
                )

                print(
                    "[GATE O-5A] finite broken =",
                    torch.isfinite(
                        broken_depth
                    ).all().item()
                )

                assert clean_nonzero > 0
                assert broken_nonzero > 0
                assert diff_nonzero > 0

                assert torch.isfinite(
                    clean_depth
                ).all()

                assert torch.isfinite(
                    broken_depth
                ).all()

                self._gate_o5a_checked = True

        else:

            # ------------------------------------------------------
            # Original Gate O-0 path
            # ------------------------------------------------------
            lidar2image = lidar2image_meta
        
        # img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        # lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
        img_feature = self.extract_img_feat(imgs, points, lidar2image,
                                            camera_intrinsics, camera2lidar,
                                            img_aug_matrix, lidar_aug_matrix,
                                            batch_input_metas)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)

        features = [img_feature, pts_feature]

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        pass

    def _build_calib_dict_from_cam2lidar(
        self,
        camera2lidar,
        camera_intrinsics,
    ):
        """Rebuild BEVFusion calibration from Camera->LiDAR extrinsic.

        Args:
            camera2lidar:
                Tensor [B, N, 4, 4]

            camera_intrinsics:
                Tensor [B, N, 4, 4]

        Returns:
            dict containing:
                cam2lidar
                cam2img
                lidar2img
        """

        # Camera -> LiDAR
        #       inverse
        # LiDAR -> Camera
        lidar2camera = torch.linalg.inv(
            camera2lidar
        )

        # Camera intrinsic @ LiDAR->Camera
        #
        # Official v1.1.0 loader represents cam2img
        # as homogeneous 4x4 matrix.
        lidar2image = torch.matmul(
            camera_intrinsics,
            lidar2camera,
        )

        return {
            'cam2lidar': camera2lidar,
            'cam2img': camera_intrinsics,
            'lidar2img': lidar2image,
        }    
    
    def _project_lidar_to_sparse_depth(
        self,
        points,
        lidar2image,
        img_aug_matrix,
        lidar_aug_matrix,
        image_hw,
    ):
        """Project LiDAR points to sparse depth maps.

        Args:
            points:
                list length B
                each tensor [P, >=3]

            lidar2image:
                [B, N, 4, 4]

            img_aug_matrix:
                [B, N, 4, 4]

            lidar_aug_matrix:
                [B, 4, 4] or [B, N, 4, 4]

            image_hw:
                (H, W)

        Returns:
            depth:
                [B, N, 1, H, W]
        """

        H, W = image_hw

        device = points[0].device
        dtype = points[0].dtype

        # ======================================================
        # Safety: unify geometry tensors on the same GPU/device.
        # ======================================================

        if not torch.is_tensor(lidar2image):
            lidar2image = torch.as_tensor(
                np.asarray(lidar2image),
                device=device,
                dtype=dtype,
            )
        else:
            lidar2image = lidar2image.to(
                device=device,
                dtype=dtype,
            )

        if not torch.is_tensor(img_aug_matrix):
            img_aug_matrix = torch.as_tensor(
                np.asarray(img_aug_matrix),
                device=device,
                dtype=dtype,
            )
        else:
            img_aug_matrix = img_aug_matrix.to(
                device=device,
                dtype=dtype,
            )

        if not torch.is_tensor(lidar_aug_matrix):
            lidar_aug_matrix = torch.as_tensor(
                np.asarray(lidar_aug_matrix),
                device=device,
                dtype=dtype,
            )
        else:
            lidar_aug_matrix = lidar_aug_matrix.to(
                device=device,
                dtype=dtype,
            )

        B = len(points)
        N = lidar2image.shape[1]

        depth = points[0].new_zeros(
            (B, N, 1, H, W)
        )

        for b in range(B):

            # --------------------------------------------------
            # Current LiDAR points
            # --------------------------------------------------
            xyz = points[b][:, :3].clone()

            # --------------------------------------------------
            # Undo LiDAR data augmentation.
            #
            # Official DepthLSSTransform does the same before
            # projecting points to the camera.
            # --------------------------------------------------
            cur_lidar_aug = lidar_aug_matrix[b]

            if cur_lidar_aug.ndim == 3:
                cur_lidar_aug = cur_lidar_aug[0]

            xyz = (
                xyz
                - cur_lidar_aug[:3, 3]
            )

            xyz = (
                torch.linalg.inv(
                    cur_lidar_aug[:3, :3]
                )
                @ xyz.transpose(0, 1)
            )

            # xyz: [3, P]

            # --------------------------------------------------
            # LiDAR -> image
            # --------------------------------------------------
            cur_l2i = lidar2image[b]

            proj = (
                cur_l2i[:, :3, :3] @ xyz
            )

            proj = (
                proj
                + cur_l2i[:, :3, 3]
                .reshape(N, 3, 1)
            )

            # Camera Z / depth
            dist = proj[:, 2, :].clone()

            # Positive depth only
            valid_z = dist > 1e-5

            z_safe = torch.clamp(
                proj[:, 2:3, :],
                min=1e-5,
                max=1e5,
            )

            proj[:, :2, :] /= z_safe

            # --------------------------------------------------
            # Apply image augmentation
            # --------------------------------------------------
            cur_img_aug = img_aug_matrix[b]

            proj = (
                cur_img_aug[:, :3, :3]
                @ proj
            )

            proj = (
                proj
                + cur_img_aug[:, :3, 3]
                .reshape(N, 3, 1)
            )

            uv = proj[:, :2, :].transpose(1, 2)

            # [N, P, 2] = x,y

            for cam in range(N):

                x = uv[cam, :, 0]
                y = uv[cam, :, 1]

                valid = (
                    (x >= 0)
                    & (x < W)
                    & (y >= 0)
                    & (y < H)
                )

                px = x[valid].long()
                py = y[valid].long()

                d = dist[cam, valid]

                depth[
                    b,
                    cam,
                    0,
                    py,
                    px,
                ] = d

        return depth