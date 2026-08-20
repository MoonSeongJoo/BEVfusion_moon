import numpy as np
import torch
import torch.nn as nn

from torchvision.models import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


class ResnetEncoder(nn.Module):

    def __init__(
        self,
        num_layers,
        pretrained,
        frozen=False,
    ):
        super().__init__()

        self.num_ch_enc = np.array(
            [64, 64, 128, 256, 512]
        )

        resnets = {
            18: (
                resnet18,
                ResNet18_Weights.DEFAULT,
            ),
            34: (
                resnet34,
                ResNet34_Weights.DEFAULT,
            ),
            50: (
                resnet50,
                ResNet50_Weights.DEFAULT,
            ),
            101: (
                resnet101,
                ResNet101_Weights.DEFAULT,
            ),
            152: (
                resnet152,
                ResNet152_Weights.DEFAULT,
            ),
        }

        if num_layers not in resnets:
            raise ValueError(
                f"Unsupported ResNet depth: "
                f"{num_layers}"
            )

        func, weights = resnets[
            num_layers
        ]

        if not pretrained:
            weights = None

        self.encoder: ResNet = func(
            weights=weights
        )

        if hasattr(self.encoder, 'fc'):
            self.encoder.fc.requires_grad_(False)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if frozen:
            self.encoder.requires_grad_(
                False
            )

    def forward(self, x):

        features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        features.append(
            self.encoder.relu(x)
        )

        features.append(
            self.encoder.maxpool(
                features[-1]
            )
        )

        features.append(
            self.encoder.layer1(
                features[-1]
            )
        )

        features.append(
            self.encoder.layer2(
                features[-1]
            )
        )

        features.append(
            self.encoder.layer3(
                features[-1]
            )
        )

        features.append(
            self.encoder.layer4(
                features[-1]
            )
        )

        return features