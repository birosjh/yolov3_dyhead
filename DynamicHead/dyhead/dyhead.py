import torch
import torch.nn.functional as F
from torch import nn

from .deform import ModulatedDeformConv
from .dyrelu import h_sigmoid, DYReLU


class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=15, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            offset_mask = self.offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))                
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x


class DyHead(nn.Module):
    def __init__(self, backbone):
        super(DyHead, self).__init__()
        self.backbone = backbone
        self.training = backbone.training
        self.hyperparams = backbone.hyperparams
        self.yolo_layers = backbone.yolo_layers
        self.seen = backbone.seen

        in_channels = int(self.hyperparams["dy_out_channels"])
        channels = int(self.hyperparams["dy_channels"])
        num_convs = int(self.hyperparams["dy_num_convs"])

        dyhead_tower = []
        for i in range(num_convs):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=Conv3x3Norm,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        # self._out_feature_strides = self.backbone._out_feature_strides
        # self._out_features = list(self._out_feature_strides.keys())
        # self._out_feature_channels = {k: channels for k in self._out_features}
        # self._size_divisibility = list(self._out_feature_strides.values())[-1]

    # @property
    # def size_divisibility(self):
    #     return self._size_divisibility

    def forward(self, x):
        x, img_size = self.backbone(x)

        layers = ["level" + str(idx) for idx in range(len(x))]

        x.reverse()
        x = dict(zip(layers, x))

        output = self.dyhead_tower(x)

        layers.reverse()
        dyhead_tower = []

        for idx, layer in enumerate(layers):
            layer_output = output[layer]
            dyhead_tower.append(
                self.yolo_layers[idx](layer_output, img_size)
            )


        return dyhead_tower if self.training else torch.cat(dyhead_tower, 1)