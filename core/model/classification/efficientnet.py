from hydra import compose, initialize
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import model_zoo

from core.model.utils import drop_connect
from core.model.utils import round_filters, round_repeats
from core.model.utils import Conv2dSamePadding


class MBConvBlock(nn.Module):
    """
    MobileNet v3 Inverted Residual Bottleneck Block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_rate, se_rate, drop_connect_rate,
                 batch_norm_eps, batch_norm_momentum):
        super().__init__()

        self.expand = (expansion_rate != 1.0)
        expand_channels = in_channels * expansion_rate

        self.se = (se_rate is not None) and (0 < se_rate <= 1)

        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        # Expansion phase
        if self.expand:
            self.pw_expand_conv = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(expand_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

        # Depthwise convolution phase
        self.dw_conv = Conv2dSamePadding(expand_channels, expand_channels, kernel_size=kernel_size,
                                         stride=stride, groups=expand_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

        # Squeeze and Excite
        if self.se:
            squeeze_channels = max(1, int(in_channels * se_rate))
            self.se_reduce = nn.Conv2d(expand_channels, squeeze_channels, kernel_size=1, stride=1, bias=True)
            self.se_expand = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, stride=1, bias=True)

        # Pointwise convolution phase (Linear Bottleneck)
        self.pw_projection_conv = nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

        # Swish activation function
        self.swish = nn.SiLU(inplace=False)

    def forward(self, x):
        inputs = x

        # Expansion forward phase
        if self.expand:
            x = self.pw_expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)

        # Depthwise forward phase
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.swish(x)

        if self.se:
            x_squeezed = F.adaptive_avg_pool2d(x, output_size=1)  # self.avg_pool(x)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.swish(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Bottleneck forward phase
        x = self.pw_projection_conv(x)
        x = self.bn2(x)

        # Inverted residual connection if possible and drop it with some probability
        if self.residual_connection:
            x = inputs + drop_connect(x, self.drop_connect_rate, self.training)
        return x


class EfficientNet(nn.Module):
    # net_param = {
    #     'b0': {'width_multiplier': 1.0, 'depth_multiplier': 1.0, 'resolution': 224, 'dropout': 0.2},
    #     'b1': {'width_multiplier': 1.0, 'depth_multiplier': 1.1, 'resolution': 240, 'dropout': 0.2},
    #     'b2': {'width_multiplier': 1.1, 'depth_multiplier': 1.2, 'resolution': 260, 'dropout': 0.3},
    #     'b3': {'width_multiplier': 1.2, 'depth_multiplier': 1.4, 'resolution': 300, 'dropout': 0.3},
    #     'b4': {'width_multiplier': 1.4, 'depth_multiplier': 1.8, 'resolution': 380, 'dropout': 0.4},
    #     'b5': {'width_multiplier': 1.6, 'depth_multiplier': 2.2, 'resolution': 456, 'dropout': 0.4},
    #     'b6': {'width_multiplier': 1.8, 'depth_multiplier': 2.6, 'resolution': 528, 'dropout': 0.5},
    #     'b7': {'width_multiplier': 2.0, 'depth_multiplier': 3.1, 'resolution': 600, 'dropout': 0.5},
    # }

    def __init__(self, name: str, num_classes=1000, weights=None, pretrained=True, load_classifier=True):
        super().__init__()

        with initialize(version_base=None, config_path="../../../config/backbone"):
            config = compose(config_name=name)
        if num_classes is not None:
            config.classifier.num_classes = num_classes

        self.name = name
        net_param = config.param
        bn_param = config.param.batch_norm
        self.block_args = config.stages
        stem_channels = config.stages.stage_1.in_channels
        classifier = config.classifier
        num_features = config.classifier.num_features

        # Apply width multiplier
        width_multiplier = net_param.width_multiplier
        if width_multiplier != 1.0:
            stem_channels = round_filters(stem_channels * width_multiplier)
            num_features = round_filters(num_features * width_multiplier)
            for stage in self.block_args.values():
                stage.in_channels = round_filters(stage.in_channels * width_multiplier)
                stage.out_channels = round_filters(stage.out_channels * width_multiplier)

        # Apply depth multiplier
        depth_multiplier = net_param.depth_multiplier
        if depth_multiplier != 1.0:
            for stage in self.block_args.values():
                stage.repeats = round_repeats(stage.repeats * depth_multiplier)

        # Total blocks after applied depth multiplier.
        # Use for drop connection scaling
        num_blocks = sum(stage.repeats for stage in self.block_args.values())

        # Stem
        self.conv_stem = Conv2dSamePadding(net_param.in_channels, stem_channels, kernel_size=3,
                                           stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(stem_channels, eps=bn_param.eps, momentum=bn_param.momentum)

        # MobileNet v3 blocks
        self.blocks = nn.ModuleList([])
        for stage in self.block_args.values():
            drop_rate = net_param.drop_connect_rate * len(self.blocks) / num_blocks
            self.blocks.append(
                MBConvBlock(
                    in_channels=stage.in_channels,
                    out_channels=stage.out_channels,
                    kernel_size=stage.kernel_size,
                    stride=stage.stride,
                    expansion_rate=stage.expansion_rate,
                    se_rate=stage.se_rate,
                    drop_connect_rate=drop_rate,
                    batch_norm_eps=bn_param.eps,
                    batch_norm_momentum=bn_param.momentum
                )
            )
            for _ in range(stage['repeats'] - 1):
                drop_rate = net_param.drop_connect_rate * len(self.blocks) / num_blocks
                self.blocks.append(
                    MBConvBlock(
                        in_channels=stage.out_channels,
                        out_channels=stage.out_channels,
                        kernel_size=stage.kernel_size,
                        stride=1,
                        expansion_rate=stage.expansion_rate,
                        se_rate=stage.se_rate,
                        drop_connect_rate=drop_rate,
                        batch_norm_eps=bn_param.eps,
                        batch_norm_momentum=bn_param.momentum
                    )
                )

        # Head
        self.conv_head = Conv2dSamePadding(
            in_channels=self.block_args.stage_7.out_channels,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features, eps=bn_param.eps, momentum=bn_param.momentum)

        # Classifier
        self.dropout = nn.Dropout(p=classifier.dropout)
        self.classifier = nn.Linear(in_features=num_features, out_features=classifier.num_classes)

        # Swish activation function
        self.swish = nn.SiLU(inplace=False)

        # Load pretrained weights
        if weights is not None or pretrained:
            self.load_pretrained_weights(weights, load_classifier)

    def forward(self, inputs):
        # Stem
        x = self.conv_stem(inputs)
        x = self.bn0(x)
        x = self.swish(x)

        # Blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.conv_head(x)
        x = self.bn1(x)
        x = self.swish(x)

        # Classifier
        x = F.adaptive_avg_pool2d(x, output_size=1)  # self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def extract_levels(self, inputs, required_levels: List[int]):
        # Stem
        x = self.conv_stem(inputs)
        x = self.bn0(x)
        x = self.swish(x)

        levels = []  # Output levels
        current_level = 1  # Index of the current level
        for block in self.blocks:
            if block.dw_conv.stride == (2, 2):
                if current_level in required_levels:
                    levels.append(x)
                current_level += 1
            x = block(x)

        if current_level in required_levels:
            levels.append(x)

        return levels

    def load_pretrained_weights(self, weights: str, load_classifier: bool):
        links = {
            'efficientnet_b0': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b0.pth',
            'efficientnet_b1': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b1.pth',
            'efficientnet_b2': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b2.pth',
            'efficientnet_b3': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b3.pth',
            'efficientnet_b4': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b4.pth',
            'efficientnet_b5': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b5.pth',
            'efficientnet_b6': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b6.pth',
            'efficientnet_b7': 'https://github.com/AlexeyVision/EfficientDet_Segmentation/releases/download/backbone_weights_v1.0/efficientnet_b7.pth',
        }
        if weights is not None:
            state_dict = torch.load(weights)
        else:
            state_dict = model_zoo.load_url(links[self.name])

        if load_classifier:
            self.load_state_dict(state_dict)
        else:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
            ret = self.load_state_dict(state_dict, strict=False)
            assert ret.missing_keys == ['classifier.weight', 'classifier.bias'], \
                'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
