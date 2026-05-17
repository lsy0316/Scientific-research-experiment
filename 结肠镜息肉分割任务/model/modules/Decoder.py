import torch
import torch.nn as nn
import torch.nn.functional as F
from module.BaseBlock import BaseConv2d, ChannelAttention


class RorD_Decoder(nn.Module):
    def __init__(self, in_high, in_low, out):
        super(RorD_Decoder, self).__init__()
        self.conv1 = BaseConv2d(in_high + in_low, out, kernel_size=3, padding=1)
        self.conv2 = BaseConv2d(out, out, kernel_size=3, padding=1)

    def forward(self, high, low):
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([high, low], dim=1)
        out = self.conv1(cat)
        out = self.conv2(out)
        return out


class IGF(nn.Module):
    def __init__(self, before_ch, r_ch, d_ch, out_ch, up=True):
        super(IGF, self).__init__()
        self.up = up
        self.conv1 = BaseConv2d(r_ch + d_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(before_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv_reduce = BaseConv2d(out_ch * 2, out_ch, kernel_size=1)
        self.ca = ChannelAttention(out_ch)
        self.conv_k = BaseConv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.conv3 = BaseConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.conv4 = BaseConv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, fea_before, fea_r, fea_d):
        if fea_before.shape[2:] != fea_r.shape[2:]:
            fea_before = F.interpolate(fea_before, size=fea_r.shape[2:], mode='bilinear', align_corners=True)
        fea_mix = self.conv1(torch.cat((fea_r, fea_d), dim=1))
        fea_before_conv = self.conv2(fea_before)

        fea_cat_reduce = self.conv_reduce(torch.cat((fea_before_conv, fea_mix), dim=1))
        fea_cat_reduce_ca = fea_cat_reduce.mul(self.ca(fea_cat_reduce)) + fea_cat_reduce
        p_block = torch.sigmoid(self.conv_k(fea_cat_reduce_ca))
        one_block = torch.ones_like(p_block)

        fea_out = fea_before_conv * (one_block - p_block) + fea_mix * p_block
        fea_out = self.relu(self.bn(fea_out))
        fea_out = self.conv3(fea_out)
        fea_out = self.conv4(fea_out)
        if self.up:
            fea_out = F.interpolate(fea_out, scale_factor=2, mode='bilinear', align_corners=True)
        return fea_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ch = [64, 128, 256, 512]   # internal channels

        # RGB decoder
        self.rgb_dec3 = RorD_Decoder(ch[3], ch[2], ch[2])
        self.rgb_dec2 = RorD_Decoder(ch[2], ch[1], ch[1])
        self.rgb_dec1 = RorD_Decoder(ch[1], ch[0], ch[0])
        self.rgb_out = nn.Sequential(
            BaseConv2d(ch[0], 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        # Depth decoder
        self.depth_dec3 = RorD_Decoder(ch[3], ch[2], ch[2])
        self.depth_dec2 = RorD_Decoder(ch[2], ch[1], ch[1])
        self.depth_dec1 = RorD_Decoder(ch[1], ch[0], ch[0])
        self.depth_out = nn.Sequential(
            BaseConv2d(ch[0], 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        # RGBD decoder (IGF)
        self.rgbd_dec3 = IGF(before_ch=ch[3], r_ch=ch[2], d_ch=ch[2], out_ch=ch[2], up=True)
        self.rgbd_dec2 = IGF(before_ch=ch[2], r_ch=ch[1], d_ch=ch[1], out_ch=ch[1], up=True)
        self.rgbd_dec1 = IGF(before_ch=ch[1], r_ch=ch[0], d_ch=ch[0], out_ch=ch[0], up=True)

        # Final prediction + edge
        self.rgbd_out = nn.Sequential(
            BaseConv2d(ch[0], 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        # Multi-scale edge heads
        self.edge_conv1 = nn.Sequential(BaseConv2d(ch[0], 32, 3, padding=1), nn.Conv2d(32, 1, 3, padding=1))
        self.edge_conv2 = nn.Sequential(BaseConv2d(ch[1], 32, 3, padding=1), nn.Conv2d(32, 1, 3, padding=1))
        self.edge_conv3 = nn.Sequential(BaseConv2d(ch[2], 32, 3, padding=1), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, rgb_feats, depth_feats, rgbd_refined, shallow_rgb=None, shallow_depth=None):
        # RGB path
        rgb_d3 = self.rgb_dec3(rgb_feats[3], rgb_feats[2])
        rgb_d2 = self.rgb_dec2(rgb_d3, rgb_feats[1])
        rgb_d1 = self.rgb_dec1(rgb_d2, rgb_feats[0])
        rgb_map = self.rgb_out(rgb_d1)

        # Depth path
        depth_d3 = self.depth_dec3(depth_feats[3], depth_feats[2])
        depth_d2 = self.depth_dec2(depth_d3, depth_feats[1])
        depth_d1 = self.depth_dec1(depth_d2, depth_feats[0])
        depth_map = self.depth_out(depth_d1)

        # RGBD path
        rgbd_refined_up = F.interpolate(rgbd_refined, size=rgb_d3.shape[2:], mode='bilinear', align_corners=True)
        rgbd_d3 = self.rgbd_dec3(rgbd_refined_up, rgb_d3, depth_d3)
        rgbd_d2 = self.rgbd_dec2(rgbd_d3, rgb_d2, depth_d2)
        rgbd_d1 = self.rgbd_dec1(rgbd_d2, rgb_d1, depth_d1)

        # Small object enhancement: inject shallow high-res feature
        if shallow_rgb is not None:
            # shallow_rgb: [B,64,H/4,W/4] (same size as rgbd_d1)
            shallow_rgb_up = F.interpolate(shallow_rgb, size=rgbd_d1.shape[2:], mode='bilinear')
            # simple additive skip connection
            rgbd_d1 = rgbd_d1 + 0.3 * shallow_rgb_up

        # Edge feature integration (sharpen segmentation)
        edge_map1 = self.edge_conv1(rgbd_d1)   # H/4
        edge_act = torch.sigmoid(edge_map1)
        rgbd_d1 = rgbd_d1 * edge_act + rgbd_d1   # modulate with edge

        # Final prediction
        rgbd_map = self.rgbd_out(rgbd_d1)

        # Multi-scale edges for loss
        edge_map2 = self.edge_conv2(rgbd_d2)
        edge_map3 = self.edge_conv3(rgbd_d3)

        return rgb_map, depth_map, rgbd_map, edge_map1, edge_map2, edge_map3