import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.VMamba.vmamba import Backbone_VSSM
from module.cmWR import cmWR
from module.BaseBlock import BaseConv2d, SpatialAttention, ChannelAttention
from module.Decoder import Decoder


class CIRNet_VMamba(nn.Module):
    def __init__(self, vmamba_cfg='tiny', pretrained_path=None):
        super(CIRNet_VMamba, self).__init__()

        # VMamba config
        if vmamba_cfg == 'tiny':
            depths = [2, 2, 9, 2]
            dims = [96, 192, 384, 768]
            drop_path_rate = 0.2
        elif vmamba_cfg == 'small':
            depths = [2, 2, 27, 2]
            dims = [96, 192, 384, 768]
            drop_path_rate = 0.3
        elif vmamba_cfg == 'base':
            depths = [2, 2, 27, 2]
            dims = [128, 256, 512, 1024]
            drop_path_rate = 0.6
        else:
            raise ValueError

        self.rgb_backbone = Backbone_VSSM(
            depths=depths, dims=dims, drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3), norm_layer='ln2d', channel_first=True, forward_type='v05'
        )
        self.depth_backbone = Backbone_VSSM(
            depths=depths, dims=dims, drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3), norm_layer='ln2d', channel_first=True, forward_type='v05'
        )

        if pretrained_path:
            state = torch.load(pretrained_path, map_location='cpu')
            self.rgb_backbone.load_state_dict(state, strict=False)
            self.depth_backbone.load_state_dict(state, strict=False)
            print(f"Loaded VMamba from {pretrained_path}")

        # Adapt to unified channels [64,128,256,512]
        self.adapt0 = BaseConv2d(dims[0], 64, kernel_size=1)
        self.adapt1 = BaseConv2d(dims[1], 128, kernel_size=1)
        self.adapt2 = BaseConv2d(dims[2], 256, kernel_size=1)
        self.adapt3 = BaseConv2d(dims[3], 512, kernel_size=1)

        # Depth quality gate (simple)
        self.depth_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # PAI units
        self.conv_fuse12 = BaseConv2d(128*2, 128, kernel_size=1)
        self.sa12 = SpatialAttention(7)
        self.conv_fuse23 = BaseConv2d(256*2, 256, kernel_size=1)
        self.sa23 = SpatialAttention(7)
        self.conv3 = BaseConv2d(512*2, 512, kernel_size=1)

        # smAR
        self.ca_rgb = ChannelAttention(512)
        self.ca_depth = ChannelAttention(512)
        self.ca_rgbd = ChannelAttention(512)
        self.sa_rgb = SpatialAttention(7)
        self.sa_depth = SpatialAttention(7)
        self.sa_rgbd = SpatialAttention(7)

        self.cmWR = cmWR(512, squeeze_ratio=1)

        self.conv_rgb = BaseConv2d(512, 512, kernel_size=3, padding=1)
        self.conv_depth = BaseConv2d(512, 512, kernel_size=3, padding=1)
        self.conv_rgbd = BaseConv2d(512, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()

    def forward(self, rgb, depth):
        depth_3ch = torch.cat([depth, depth, depth], dim=1)

        rgb_stages = self.rgb_backbone(rgb)      # [stage0,stage1,stage2,stage3]
        depth_stages = self.depth_backbone(depth_3ch)

        f0_r = self.adapt0(rgb_stages[0])
        f1_r = self.adapt1(rgb_stages[1])
        f2_r = self.adapt2(rgb_stages[2])
        f3_r = self.adapt3(rgb_stages[3])

        f0_d = self.adapt0(depth_stages[0])
        f1_d = self.adapt1(depth_stages[1])
        f2_d = self.adapt2(depth_stages[2])
        f3_d = self.adapt3(depth_stages[3])

        # Apply depth quality gate before fusion
        depth_weight = self.depth_quality(f3_d)
        f3_d = f3_d * depth_weight   # weight highest stage only; for simplicity

        # PAI
        rgbd1 = self.conv_fuse12(torch.cat([f1_r, f1_d], dim=1))
        att1 = self.sa12(rgbd1)
        att1_down = F.interpolate(att1, size=f2_r.shape[2:], mode='bilinear')

        rgbd2 = self.conv_fuse23(torch.cat([f2_r, f2_d], dim=1))
        rgbd2 = rgbd2 * att1_down + rgbd2
        att2 = self.sa23(rgbd2)
        att2_down = F.interpolate(att2, size=f3_r.shape[2:], mode='bilinear')

        rgbd3 = self.conv3(torch.cat([f3_r, f3_d], dim=1))
        rgbd3 = rgbd3 * att2_down + rgbd3

        # smAR
        B, C, H, W = f3_r.shape
        P = H * W
        rgb_sa = self.sa_rgb(f3_r).view(B, -1, P)
        depth_sa = self.sa_depth(f3_d).view(B, -1, P)
        rgbd_sa = self.sa_rgbd(rgbd3).view(B, -1, P)

        rgb_ca = self.ca_rgb(f3_r).view(B, C, -1)
        depth_ca = self.ca_depth(f3_d).view(B, C, -1)
        rgbd_ca = self.ca_rgbd(rgbd3).view(B, C, -1)

        rgb_att = torch.bmm(rgb_ca, rgb_sa).view(B, C, H, W)
        depth_att = torch.bmm(depth_ca, depth_sa).view(B, C, H, W)
        rgbd_att = torch.bmm(rgbd_ca, rgbd_sa).view(B, C, H, W)

        rgb_smAR = f3_r * rgb_att + f3_r
        depth_smAR = f3_d * depth_att + f3_d
        rgbd_smAR = rgbd3 * rgbd_att + rgbd3

        rgb_smAR = self.conv_rgb(rgb_smAR)
        depth_smAR = self.conv_depth(depth_smAR)
        rgbd_smAR = self.conv_rgbd(rgbd_smAR)

        # cmWR
        rgb_cmWR, depth_cmWR, rgbd_cmWR = self.cmWR(rgb_smAR, depth_smAR, rgbd_smAR)

        # Decoder - also pass shallow features for small object enhancement
        rgb_feats = [f0_r, f1_r, f2_r, f3_r]
        depth_feats = [f0_d, f1_d, f2_d, f3_d]
        rgb_map, depth_map, rgbd_map, edge_map1, edge_map2, edge_map3 = self.decoder(
            rgb_feats, depth_feats, rgbd_cmWR, shallow_rgb=f0_r, shallow_depth=f0_d
        )
        return rgb_map, depth_map, rgbd_map, edge_map1, edge_map2, edge_map3