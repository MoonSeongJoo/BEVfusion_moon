# mmdet3d/models/data_preprocessors/custom_data_preprocessor.py

import math
from typing import List, Tuple

import numpy as np
import torch
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor

@MODELS.register_module()
class CustomDet3DDataPreprocessor(Det3DDataPreprocessor):
    """
    Custom Data Preprocessor that handles 5D tensor inputs (NVCHW) for 
    multi-view images and preserves all original keys from the input dictionary.
    """

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """
        Overrides simple_process to preserve additional keys like 'img_original'
        and 'points_original' from the input data, while also handling
        voxelization and image processing.
        """
        # Get padding shape before data collation
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        # Collate data and move to the target device
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        #
        # --- KEY FIX ---
        # Copy the original `inputs` dictionary to preserve all keys,
        # including 'img_original', 'points_original', etc.
        batch_inputs = inputs.copy()
        #
        
        # Voxelize point cloud if enabled
        if 'points' in inputs and self.voxel:
            voxel_dict = self.voxelize(inputs['points'], data_samples)
            batch_inputs['voxels'] = voxel_dict

        # Process image data
        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            # Apply batch-level augmentations if any
            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            
            # Update 'imgs' in batch_inputs with the processed (and possibly augmented) version
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[Tuple[int, int]]:
        """
        Get the pad_shape of each image. Handles 5D (NVCHW) tensor inputs.
        """
        _batch_inputs = data['inputs']['img']
        # Handle list of tensors (pseudo_collate)
        if is_seq_of(_batch_inputs, torch.Tensor):
            return super()._get_pad_shape(data)
        
        # Handle batched tensor (default_collate)
        elif isinstance(_batch_inputs, torch.Tensor):
            if _batch_inputs.dim() == 5:  # Handle NVCHW
                h, w = _batch_inputs.shape[3:]
                pad_h = int(np.ceil(h / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(w / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
            elif _batch_inputs.dim() == 4: # Original logic for NCHW
                h, w = _batch_inputs.shape[2:]
                pad_h = int(np.ceil(h / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(w / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
            else:
                raise AssertionError(
                    'Input image tensor must be 4D (NCHW) or 5D (NVCHW), '
                    f'but got a tensor with shape: {_batch_inputs.shape}')
            return batch_pad_shape
        else:
            raise TypeError(f'Unsupported type for image inputs: {type(data)}')

    def collate_data(self, data: dict) -> dict:
        """
        Collates data samples into a batch. Handles 5D (NVCHW) tensor inputs.
        """
        data = self.cast_data(data)

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']

            if is_seq_of(_batch_imgs, torch.Tensor):
                return super().collate_data(data)
            
            elif isinstance(_batch_imgs, torch.Tensor):
                if _batch_imgs.dim() == 5: # Handle NVCHW (N, V, C, H, W)
                    num_samples, num_views, C, H, W = _batch_imgs.shape
                    imgs_4d = _batch_imgs.reshape(num_samples * num_views, C, H, W)

                    if self._channel_conversion:
                        imgs_4d = imgs_4d[:, [2, 1, 0], ...]
                    
                    imgs_4d = imgs_4d.float()
                    
                    if self._enable_normalize:
                        imgs_4d = (imgs_4d - self.mean) / self.std

                    processed_imgs = imgs_4d.reshape(num_samples, num_views, C, H, W)

                    target_h = math.ceil(H / self.pad_size_divisor) * self.pad_size_divisor
                    target_w = math.ceil(W / self.pad_size_divisor) * self.pad_size_divisor
                    pad_h = target_h - H
                    pad_w = target_w - W
                    
                    batch_imgs = F.pad(processed_imgs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)

                elif _batch_imgs.dim() == 4: # Original logic for NCHW
                    batch_imgs = super().preprocess_img(_batch_imgs)
                    h, w = batch_imgs.shape[2:]
                    target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                    target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                    pad_h = target_h - h
                    pad_w = target_w - w
                    batch_imgs = F.pad(batch_imgs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
                else:
                    raise AssertionError(
                        'Input image tensor must be 4D (NCHW) or 5D (NVCHW), '
                        f'but got a tensor with shape: {_batch_imgs.shape}')

                # Rename 'img' to 'imgs' after processing
                data['inputs']['imgs'] = batch_imgs
                if 'img' in data['inputs']:
                    del data['inputs']['img']

            else:
                 raise TypeError(f'Unsupported type for image inputs: {type(data)}')
        
        data.setdefault('data_samples', None)
        return data