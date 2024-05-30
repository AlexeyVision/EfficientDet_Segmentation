import math
import torch
from torch import nn
import torch.nn.functional as F


def round_filters(filters, depth_divisor=8, min_depth=8):
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return new_filters


def round_repeats(repeats):
    return int(math.ceil(repeats))


def drop_connect(inputs, drop_prob, training):
    assert 0 <= drop_prob <= 1, "Drop connect probability rate must be in range of [0, 1]"

    if not training:
        return inputs

    keep_prob = 1 - drop_prob

    # Binary tensor from Bernoulli distribution
    mask = torch.empty((inputs.shape[0], 1, 1, 1), device=inputs.device).bernoulli_(p=keep_prob)

    # Apply mask and normalization
    outputs = inputs / keep_prob * mask
    return outputs


class Conv2dSamePadding(nn.Conv2d):
    """
    2D Convolution like TensorFlow "same" mode
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        input_height = x.size(dim=-2)
        input_width = x.size(dim=-1)

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        dilation_height, dilation_width = self.dilation
        output_height = math.ceil(input_height / stride_height)
        output_width = math.ceil(input_width / stride_width)

        padding_height = max((output_height - 1) * stride_height + (kernel_height - 1) * dilation_height + 1 - input_height, 0)
        padding_width = max((output_width - 1) * stride_width + (kernel_width - 1) * dilation_width + 1 - input_width, 0)

        if padding_height > 0 or padding_width > 0:
            padding = (padding_width // 2, padding_width - padding_width // 2,
                       padding_height // 2, padding_height - padding_height // 2)
            x = F.pad(x, pad=padding, mode="constant", value=0)

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MaxPool2dSamePadding(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        input_height = x.size(dim=-2)
        input_width = x.size(dim=-1)

        kernel_height, kernel_width = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        stride_height, stride_width = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        dilation_height, dilation_width = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        output_height = math.ceil(input_height / stride_height)
        output_width = math.ceil(input_width / stride_width)

        padding_height = max((output_height - 1) * stride_height + (kernel_height - 1) * dilation_height + 1 - input_height, 0)
        padding_width = max((output_width - 1) * stride_width + (kernel_width - 1) * dilation_width + 1 - input_width, 0)

        if padding_height > 0 or padding_width > 0:
            padding = (padding_width // 2, padding_width - padding_width // 2,
                       padding_height // 2, padding_height - padding_height // 2)
            x = F.pad(x, pad=padding, mode="constant", value=0)

        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.dw_conv = Conv2dSamePadding(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                         stride=1, dilation=1, groups=in_channels, bias=False)
        self.pw_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                 stride=1, dilation=1, bias=True)


    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
