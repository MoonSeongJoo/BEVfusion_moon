import torch
import torch.nn as nn
import torch.nn.functional as F


def correlation2d(
    input1,
    input2,
    max_displacement,
):
    """Pure PyTorch correlation cost volume.

    Input:
        input1: [B, C, H, W]
        input2: [B, C, H, W]

    Output:
        [B, (2d+1)^2, H, W]
    """

    assert (
        input1.shape
        == input2.shape
    )

    _, _, H, W = input1.shape

    d = max_displacement

    input2_pad = F.pad(
        input2,
        [d, d, d, d],
    )

    cost_volumes = []

    for dy in range(
        2 * d + 1
    ):

        for dx in range(
            2 * d + 1
        ):

            shifted = input2_pad[
                :,
                :,
                dy:dy + H,
                dx:dx + W,
            ]

            cost = (
                input1
                * shifted
            ).mean(
                dim=1,
                keepdim=True,
            )

            cost_volumes.append(
                cost
            )

    return torch.cat(
        cost_volumes,
        dim=1,
    )


class Correlation(nn.Module):

    def __init__(
        self,
        pad_size=4,
        kernel_size=1,
        max_displacement=4,
        stride1=1,
        stride2=1,
        corr_multiply=1,
    ):
        super().__init__()

        # Kept for compatibility with
        # the original LCCNet API.
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = (
            max_displacement
        )

        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = (
            corr_multiply
        )

    def forward(
        self,
        input1,
        input2,
    ):

        out = correlation2d(
            input1,
            input2,
            self.max_displacement,
        )

        if self.corr_multiply != 1:
            out = (
                out
                * self.corr_multiply
            )

        return out