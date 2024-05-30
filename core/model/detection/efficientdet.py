from typing import Optional
from torch import nn
from core.model.utils import SeparableConv, MaxPool2dSamePadding, Conv2dSamePadding


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network

        P7_0 --------------------------> P7_2 ------->
             |                             ↑
             --------------                |
                          ↓                |
        P6_0 ---------> P6_1 ---------> P6_2 -------->
             |            |                ↑
             ------------------------------| 
                          ↓                ↑
        P5_0 ---------> P5_1 ---------> P5_2 -------->
             |            |                ↑
             ------------------------------| 
                          ↓                ↑
        P4_0 ---------> P4_1 ---------> P4_2 -------->
             |            |                ↑
             ------------------------------| 
                          |                ↑
                          ---------------↓ |
        P3_0 ---------> P3_1 ---------> P3_2 -------->
    
    """
    def __init__(self, num_channels: int, backbone_out_channel_sizes: Optional[list] = None,
                 batch_norm_eps=0.01, batch_norm_momentum=1e-3):
        super(BiFPN, self).__init__()

        self.p6_hidden_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P7 and P6
        self.p5_hidden_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P6 and P5
        self.p4_hidden_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P5 and P4
        self.p3_output_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P4 and P3

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_output_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P3 and P4
        self.p5_output_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P4 and P5
        self.p6_output_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P5 and P6
        self.p7_output_conv = SeparableConv(in_channels=num_channels, out_channels=num_channels)  # From P6 and P6

        self.downsample = MaxPool2dSamePadding(kernel_size=3, stride=2, dilation=1)
        # self.p4_downsample = MaxPool2dSamePadding(kernel_size=3, stride=2)
        # self.p5_downsample = MaxPool2dSamePadding(kernel_size=3, stride=2)
        # self.p5_downsample = MaxPool2dSamePadding(kernel_size=3, stride=2)
        # self.p6_downsample = MaxPool2dSamePadding(kernel_size=3, stride=2)
        # self.p7_downsample = MaxPool2dSamePadding(kernel_size=3, stride=2)

        self.swish = nn.SiLU(inplace=False)

        self.from_backbone = True if backbone_out_channel_sizes else False
        if backbone_out_channel_sizes:
            assert len(backbone_out_channel_sizes) == 3, ('The number of outputs from EfficientNet does not match the input '
                                                     'to BiFPN')
            p3_channels, p4_channels, p5_channels = backbone_out_channel_sizes

            # Input conv to create P3
            self.p3_input_conv = Conv2dSamePadding(in_channels=p3_channels, out_channels=num_channels, kernel_size=1,
                                                   stride=1, dilation=1, groups=1, bias=True)
            self.p3_bn = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

            # First input conv to create P4
            self.p4_input_conv_1 = Conv2dSamePadding(in_channels=p4_channels, out_channels=num_channels, kernel_size=1,
                                                     stride=1, dilation=1, groups=1, bias=True)
            self.p4_bn_1 = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

            # Second input conv to create P4
            self.p4_input_conv_2 = Conv2dSamePadding(in_channels=p4_channels, out_channels=num_channels, kernel_size=1,
                                                     stride=1, dilation=1, groups=1, bias=True)
            self.p4_bn_2 = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

            # First input conv to create P5
            self.p5_input_conv_1 = Conv2dSamePadding(in_channels=p5_channels, out_channels=num_channels, kernel_size=1,
                                                     stride=1, dilation=1, groups=1, bias=True)
            self.p5_bn_1 = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

            # Second input conv to create P5
            self.p5_input_conv_2 = Conv2dSamePadding(in_channels=p5_channels, out_channels=num_channels, kernel_size=1,
                                                     stride=1, dilation=1, groups=1, bias=True)
            self.p5_bn_2 = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

            # Conversion P5 to P6 with further upsample
            self.p6_input_conv = Conv2dSamePadding(in_channels=p5_channels, out_channels=num_channels, kernel_size=3,
                                                   stride=1, dilation=1, groups=1, bias=True)
            self.p6_bn = nn.BatchNorm2d(num_channels, eps=batch_norm_eps, momentum=batch_norm_momentum)

    def forward(self, inputs):
        if self.from_backbone:
            p3, p4, p5 = inputs

            p3_input = self.p3_input_conv(p3)
            p3_input = self.p3_bn(p3_input)

            p4_input = self.p4_input_conv_1(p4)
            p4_input = self.p4_bn_1(p4_input)

            p5_input = self.p5_input_conv_1(p5)
            p5_input = self.p5_bn_1(p5_input) 

            p6_input = self.p6_input_conv(p5)
            p6_input = self.p6_bn(p6_input)
            p6_input = self.downsample(p6_input)

            p7_input = self.downsample(p6_input)  # Downsample only
        else:
            p3_input, p4_input, p5_input, p6_input, p7_input = inputs

        # P7 and P6 -> P6 hidden
        p6_hidden = self.p6_hidden_conv(self.swish(p6_input + self.upsample(p7_input)))

        # P6 hidden and P5 -> P5 hidden
        p5_hidden = self.p5_hidden_conv(self.swish(p5_input + self.upsample(p6_hidden)))

        # P5 hidden and P4 -> P4 hidden
        p4_hidden = self.p4_hidden_conv(self.swish(p4_input + self.upsample(p5_hidden)))

        # P4 hidden and P3 -> P3 output
        p3_output = self.p3_output_conv(self.swish(p3_input + self.upsample(p4_hidden)))

        # Taking raw P4 and P5 inputs
        # This leads to improved convergence
        if self.from_backbone:
            p4_input = self.p4_input_conv_2(p4) 
            p4_input = self.p4_bn_2(p4_input) 
            p5_input = self.p5_input_conv_2(p5)
            p5_input = self.p5_bn_2(p5_input)

        # P4 input, P4 hidden, P3 output -> P4 output
        p4_output = self.p4_output_conv(self.swish(p4_input + p4_hidden + self.downsample(p3_output)))

        # P5 input, P5 hidden, P4 output -> P5 output
        p5_output = self.p5_output_conv(self.swish(p5_input + p5_hidden + self.downsample(p4_output)))

        # P6 input, P6 hidden, P5 output -> P6 output
        p6_output = self.p6_output_conv(self.swish(p6_input + p6_hidden + self.downsample(p5_output)))

        # P7 input and P6 output -> P7 output
        p7_output = self.p7_output_conv(self.swish(p7_input + self.downsample(p6_output)))

        return p3_output, p4_output, p5_output, p6_output, p7_output
