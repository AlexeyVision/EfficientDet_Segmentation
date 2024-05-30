from hydra import compose, initialize

import torch
from torch import nn
from core.model.utils import SeparableConv
from core.model.classification.efficientnet import EfficientNet
from core.model.detection.efficientdet import BiFPN


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_channels, num_classes, num_layers, batch_norm_eps, batch_norm_momentum):
        super().__init__()

        self.head_conv = nn.ModuleList()
        self.head_bn = nn.ModuleList()
        self.swish = nn.SiLU()

        for i in range(num_layers):
            if i == 0:
                self.head_conv.append(SeparableConv(in_channels=in_channels, out_channels=num_channels))
                self.head_bn.append(nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum))
            else:
                self.head_conv.append(SeparableConv(in_channels=num_channels))
                self.head_bn.append(nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum))

        self.classifier = SeparableConv(in_channels=num_channels, out_channels=num_classes)

    def forward(self, x):
        for conv, bn in zip(self.head_conv, self.head_bn):
            x = conv(x)
            x = bn(x)
            x = self.swish(x)
        x = self.classifier(x)
        return x


class SegmentationHeadFPN(nn.Module):
    num_layers = 4
    
    def __init__(self, in_channels, num_channels, num_classes, batch_norm_eps, batch_norm_momentum):
        super().__init__()   
        
        self.head_fpn = nn.ModuleList()
        self.head_bn = nn.ModuleList()
        self.swish = nn.SiLU()

        for i in range(SegmentationHeadFPN.num_layers):
            if i == 0:
                self.head_fpn.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_channels, kernel_size=2, stride=2))
                self.head_bn.append(nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum))
            else:
                self.head_fpn.append(nn.ConvTranspose2d(in_channels=in_channels + num_channels, out_channels=num_channels, kernel_size=2, stride=2))
                self.head_bn.append(nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum))
                
        self.classifier = nn.ConvTranspose2d(in_channels=in_channels + num_channels, out_channels=num_classes, kernel_size=2, stride=2)

    def forward(self, inputs):
        x = inputs[0]

        for conv, bn, x_connect in zip(self.head_fpn, self.head_bn, inputs[1:]):
            x = conv(x)
            x = bn(x)
            x = self.swish(x)
            x = torch.cat([x, x_connect], axis=1)
        x = self.classifier(x)
        return x


class EfficientDetSegmentation(nn.Module):
    """
    Modified EfficientDet model for segmentation task.
    Model includes EfficientNet which can be configured in the corresponding files like this model.

    Model structure:
    EfficientNet -> BiFPN -> Segmentation Head from P2 BiFPN features

    Segmentation Head uses 2 consistent deconvolutions to increase the resolution by 4 times. 
    The main source: https://arxiv.org/pdf/1911.09070
    """
    def __init__(self,
                 name: str = 'efficientdet_d4_seg',
                 num_classes=21,
                 backbone_pretrained=True,
                 backbone_weights=None,
                 weights=None
    ):
        """
        Args:
            name: model name from config/segmentation without '.yaml'
            num_classes: number of classes
            backbone_pretrained: True if you want use pretrain EfficientNet backbone
            backbone_weights: path to backbone weights, if you want to use own EfficientNet weights
            weights: path to weights of this model. In this case, you don't need to use
                     backbone_pretrained and backbone_weights argument
        Note:
            You may configure more model details in corresponding file 
        """
        super().__init__()

        with initialize(version_base=None, config_path="../../../config/segmentation"):
            config = compose(config_name=name)

        batch_norm_momentum = config.param.batch_norm.momentum
        batch_norm_eps = config.param.batch_norm.eps
        self.required_levels = config.backbone.required_levels

        self.backbone = EfficientNet(
            name=config.backbone.name,
            weights=backbone_weights if backbone_weights is not None else config.backbone.weights,
            pretrained=backbone_pretrained if backbone_pretrained is not None else config.backbone.pretrained,
            load_classifier=False
        )

        # TODO: Erase working part of the model only if you use SegmentationHead
        # del self.backbone.conv_head
        # del self.backbone.bn1
        # del self.backbone.dropout
        # del self.backbone.classifier

        self.feature_network = nn.ModuleList()
        for i in range(config.neck.num_layers):
            self.feature_network.append(
                BiFPN(
                    num_channels=config.neck.num_channels,
                    backbone_out_channel_sizes=config.neck.input_channel_sizes if i == 0 else None,
                    batch_norm_eps=batch_norm_eps,
                    batch_norm_momentum=batch_norm_momentum
                )
            )

        if hasattr(config, 'head'):
            self.headFPN = False
            self.expand_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=config.neck.num_channels, out_channels=config.neck.num_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(config.neck.num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
                nn.SiLU(),
                nn.ConvTranspose2d(in_channels=config.neck.num_channels, out_channels=config.neck.num_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(config.neck.num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
                nn.SiLU()
            )
            self.classifier = SegmentationHead(
                in_channels=config.neck.num_channels,
                num_channels=config.head.num_channels,
                num_classes=num_classes if num_classes else config.head.num_classes,
                num_layers=config.head.num_layers,
                batch_norm_eps=batch_norm_eps,
                batch_norm_momentum=batch_norm_momentum
            )
        elif hasattr(config, 'headFPN'):
            self.headFPN = True
            self.classifier = SegmentationHeadFPN(
                in_channels=config.neck.num_channels,
                num_channels=config.headFPN.num_channels,
                num_classes=num_classes if num_classes else config.headFPN.num_classes,
                batch_norm_eps=batch_norm_eps,
                batch_norm_momentum=batch_norm_momentum
            )
        
        # Load pretrained weights
        if weights:
            self.load_pretrained_weights(weights)


    def forward(self, inputs):
        x = self.backbone.extract_levels(inputs, required_levels=self.required_levels)  # Backbone

        for fpn_layer in self.feature_network:  # BiFPN
            x = fpn_layer(x)

        # Using P2 output only
        if not self.headFPN:
            x = self.expand_conv(x[0])
        else:
            x = list(reversed(x)) # P3, P4, P5, P6, P7 -> P7, P6, P5, P4, P3

        x = self.classifier(x)  # Classifier
        return x

    def load_pretrained_weights(self, weights: str):
        state_dict = torch.load(weights, map_location='cpu')

        # Remove 'module.' prefix
        # It is need when you save weights from model with torch.nn.DataParallel()
        corrected_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  
            corrected_state_dict[new_key] = value

        ret = self.load_state_dict(corrected_state_dict, strict=False)

        if ret.missing_keys:
            print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))



