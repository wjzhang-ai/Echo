import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2), DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, dilation=12
        )
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, dilation=18
        )

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode="bilinear")

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(
            torch.cat(
                [
                    image_features,
                    atrous_block1,
                    atrous_block6,
                    atrous_block12,
                    atrous_block18,
                ],
                dim=1,
            )
        )
        return net

# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)
 
# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
 
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = super(ASPPPooling, self).forward(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
 
# class ASPP(nn.Module):
#     def __init__(self, in_channels,out_channels , atrous_rates):
#         super(ASPP, self).__init__()
       
#         modules = []
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
 
#         rate1, rate2, rate3 = tuple(atrous_rates)
#         modules.append(ASPPConv(in_channels, out_channels, rate1))
#         modules.append(ASPPConv(in_channels, out_channels, rate2))
#         modules.append(ASPPConv(in_channels, out_channels, rate3))
#         modules.append(ASPPPooling(in_channels, out_channels))
 
#         self.convs = nn.ModuleList(modules)
 
#         self.project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))
 
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)



class UNet_aspp_asff(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        bilinear: bool = False,
        base_c: int = 64,
        gpu_ids=[],
    ):
        super(UNet_aspp_asff, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # self.in_conv = DoubleConv(in_channels, base_c)
        self.in_conv = Down(2, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.aspp1 = ASPP(32,32)
        # self.aspp2 = ASPP(64,64)
        # self.aspp3 = ASPP(128,128)
        # self.aspp4 = ASPP(256,256)
        self.aspp1 = ASPP(64, 64)
        self.aspp2 = ASPP(128,128,)
        self.aspp3 = ASPP(256,256,)
        self.aspp4 = ASPP(512, 512)
        self.asff_0 = ASFF(level=0)
        self.asff_1 = ASFF(level=1)
        self.asff_2 = ASFF(level=2)

        # self.aspp4 = ASPP()

    def forward(self, x):

        x1 = self.in_conv(x)
        x1 = self.aspp1(x1)  

        x2 = self.down1(x1)  
        x2 = self.aspp2(x2)  
        feture_2 = x2#[64,128,32,32]
        x3 = self.down2(x2)  

        x3 = self.aspp3(x3)  
        feture_1 = x3#[64,256,16,16]
        x4 = self.down3(x3)  # 512 16 16
        x4 = self.aspp4(x4)
        feture_0 = x4#[64,512,8,8]

        fused_features_0 = self.asff_0(feture_0, feture_1, feture_2)#[64,512,8,8]
        fused_features_1 = self.asff_1(feture_0, feture_1, feture_2)#[64,256,16,16]
        fused_features_2 = self.asff_2(feture_0, feture_1, feture_2)
        x = self.up2(fused_features_0, fused_features_1)#[64,128,32,32]

        x = self.up3(x, fused_features_2)

        return x


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=False,
        ),
    )
    stage.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module("leaky", nn.LeakyReLU(0.1))
    else:
        stage.add_module("relu6", nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(256, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 128, 3, 1)

        compress_c = (
            8 if rfb else 16
        )  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(
            compress_c * 3, 3, kernel_size=1, stride=1, padding=0
        )
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode="nearest"
            )
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode="nearest"
            )
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                level_1_compressed, scale_factor=2, mode="nearest"
            )
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1
        )
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :]
            + level_1_resized * levels_weight[:, 1:2, :, :]
            + level_2_resized * levels_weight[:, 2:, :, :]
        )

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


# if __name__ == "__main__":

#     device = torch.device("cpu")
#     input = torch.randn(1, 2, 128, 128).to(device)
#     unet_aspp = UNet_aspp_asff()
#     x = unet_aspp(input)
#     print("final", x.shape)
    # asff_0 = ASFF(level=0)
    # asff_1 = ASFF(level=1)
    # asff_2 = ASFF(level=2)
    # fused_features_0 = asff_0(x_level_0, x_level_1, x_level_2)
    # fused_features_1 = asff_1(x_level_0, x_level_1, x_level_2)
    # fused_features_2 = asff_2(x_level_0, x_level_1, x_level_2)

    # print(fused_features_0.shape, fused_features_1.shape, fused_features_2.shape)
