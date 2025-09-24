import torch

from mmdet.models.task_modules.coders import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DistancePointBBoxCoder(BaseBBoxCoder):
    """Distance BBox coder for 2D boxes.
    Args:
        base_pos (list[int]): The base point position of the bbox.
    """

    def __init__(self, base_pos=(4, 1)):
        super(BaseBBoxCoder, self).__init__()
        self.base_pos = base_pos

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """
        # "Left, Top, Right, Bottom" format.
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        base_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        base_y = bboxes[:, self.base_pos[1]]
        encoded_bboxes = torch.stack(
            [
                base_x - gt_bboxes[:, 0],  # l
                base_y - gt_bboxes[:, 1],  # t
                gt_bboxes[:, 2] - base_x,  # r
                gt_bboxes[:, 3] - base_y,  # b
            ],
            dim=-1,
        )
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes.
        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        base_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        base_y = bboxes[:, self.base_pos[1]]
        decoded_bboxes = torch.stack(
            [
                base_x - pred_bboxes[:, 0],
                base_y - pred_bboxes[:, 1],
                base_x + pred_bboxes[:, 2],
                base_y + pred_bboxes[:, 3],
            ],
            dim=-1,
        )
        return decoded_bboxes