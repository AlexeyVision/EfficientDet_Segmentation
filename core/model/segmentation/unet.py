from hydra import compose, initialize

import torch
from torch import nn
from torch.nn.functional import pad


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm_eps, batch_norm_momentum):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm_eps, batch_norm_momentum):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              batch_norm_momentum=batch_norm_momentum, batch_norm_eps=batch_norm_eps)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        connect = self.conv(x)
        out = self.pool(connect)
        return out, connect


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, deconv_kernel_size, batch_norm_eps, batch_norm_momentum):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=deconv_kernel_size, stride=2)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              batch_norm_momentum=batch_norm_momentum, batch_norm_eps=batch_norm_eps)

    def forward(self, inputs, connect):
        out = self.up(inputs)

        diff_height = connect.size()[2] - out.size()[2]
        diff_width = connect.size()[3] - out.size()[3]

        if out.shape != connect.shape:
            # Pad last dim by (diff_width // 2, diff_width - diff_width // 2)
            # and 2nd to last by (diff_height // 2, diff_height - diff_height // 2)
            p2d = (diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2)
            out = pad(out, p2d, "constant", 0)

        concat_out = torch.cat((out, connect), dim=1)
        concat_out = self.conv(concat_out)
        return concat_out


class UNet(nn.Module):
    """
    Default UNet model for segmentation task.
    This modification uses convolutions with padding.

    Model structure:
    Encoder -> Bottleneck -> Decoder

    The main source: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, in_channels, num_classes, weights=None):
        """
        Args: 
            in_channels: input image channels
            num_classes: number of classes
            weights: path to model weights
        Note:
            You may configure more model details in corresponding file
        """
        super(UNet, self).__init__()

        with initialize(version_base=None, config_path="../../../config/segmentation"):
            config = compose(config_name='unet')
        if num_classes is not None:
            config.classifier.num_classes = num_classes
        if in_channels is not None:
            config.encoder.stage_1.in_channels = in_channels
        
        # Batch norm params
        eps = config.param.batch_norm.eps
        momentum = config.param.batch_norm.momentum

        # Encoder part
        st1 = config.encoder.stage_1
        self.down1 = DownConv(in_channels=st1.in_channels, out_channels=st1.out_channels, kernel_size=st1.kernel_size,
                              batch_norm_eps=eps, batch_norm_momentum=momentum)
        st2 = config.encoder.stage_2
        self.down2 = DownConv(in_channels=st2.in_channels, out_channels=st2.out_channels, kernel_size=st2.kernel_size,
                              batch_norm_eps=eps, batch_norm_momentum=momentum)
        st3 = config.encoder.stage_3
        self.down3 = DownConv(in_channels=st3.in_channels, out_channels=st3.out_channels, kernel_size=st3.kernel_size,
                              batch_norm_eps=eps, batch_norm_momentum=momentum)
        st4 = config.encoder.stage_4
        self.down4 = DownConv(in_channels=st4.in_channels, out_channels=st4.out_channels, kernel_size=st4.kernel_size,
                              batch_norm_eps=eps, batch_norm_momentum=momentum)

        # Bottleneck part
        bt = config.bottleneck
        self.bottleneck = ConvBlock(in_channels=bt.in_channels, out_channels=bt.out_channels, kernel_size=bt.kernel_size,
                                    batch_norm_eps=eps, batch_norm_momentum=momentum)

        # Decoder part
        st1 = config.decoder.stage_1
        self.up1 = UpConv(in_channels=st1.in_channels, out_channels=st1.out_channels,
                          kernel_size=st1.kernel_size, deconv_kernel_size=st1.deconv_kernel_size,
                          batch_norm_eps=eps, batch_norm_momentum=momentum)
        st2 = config.decoder.stage_2
        self.up2 = UpConv(in_channels=st2.in_channels, out_channels=st2.out_channels, 
                          kernel_size=st2.kernel_size, deconv_kernel_size=st2.deconv_kernel_size,
                          batch_norm_eps=eps, batch_norm_momentum=momentum)
        st3 = config.decoder.stage_3
        self.up3 = UpConv(in_channels=st3.in_channels, out_channels=st3.out_channels, 
                          kernel_size=st3.kernel_size, deconv_kernel_size=st3.deconv_kernel_size,
                          batch_norm_eps=eps, batch_norm_momentum=momentum)
        st4 = config.decoder.stage_4
        self.up4 = UpConv(in_channels=st4.in_channels, out_channels=st4.out_channels, 
                          kernel_size=st4.kernel_size, deconv_kernel_size=st4.deconv_kernel_size,
                          batch_norm_eps=eps, batch_norm_momentum=momentum)

        # Classifier
        out = config.classifier
        self.outputs = nn.Conv2d(in_channels=out.in_channels, out_channels=out.num_classes, kernel_size=1)

        # Load pretrained weights
        if weights is not None:
            self.load_pretrained_weights(weights)

    def forward(self, x):
        out, connect1 = self.down1(x)
        out, connect2 = self.down2(out)
        out, connect3 = self.down3(out)
        out, connect4 = self.down4(out)

        out = self.bottleneck(out)

        out = self.up1(out, connect4)
        out = self.up2(out, connect3)
        out = self.up3(out, connect2)
        out = self.up4(out, connect1)

        out = self.outputs(out)
        return out

    def load_pretrained_weights(self, weights: str):
        state_dict = torch.load(weights)
        ret = self.load_state_dict(state_dict, strict=False)

        if ret.missing_keys:
            print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
