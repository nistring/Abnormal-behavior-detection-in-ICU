import torch.nn as nn
from .builder import SPPE
from .layers.Resnet import ResNet
import torch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

class ConvBlock(nn.Module):
    def __init__(self, num_features_in, num_classes=15, feature_size=256):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()

        self.deconv = self._make_deconv_layer(feature_size)

        self.final_layer = nn.Conv2d(feature_size, num_classes, kernel_size=1, stride=1, padding=0)

    def _make_deconv_layer(self, feature_size):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = nn.BatchNorm2d(feature_size)
        deconv2 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = nn.BatchNorm2d(feature_size)
        deconv3 = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = nn.BatchNorm2d(feature_size)

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.deconv(x)
        x = self.final_layer(x)

        # Channel last
        return x


@SPPE.register_module
class A2J(nn.Module):
    def __init__(self, **cfg):
        super(A2J, self).__init__()

        self._preset_cfg = cfg["PRESET"]

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm  # noqa: F401,F403

        assert cfg["NUM_LAYERS"] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {
            k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()
        }
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.sum_anchors = Summer(PositionalEncoding2D(self._preset_cfg["NUM_JOINTS"]))
        self.regressionModel = ConvBlock(2048, num_classes=self._preset_cfg["NUM_JOINTS"])
        self.classificationModel = ConvBlock(2048, num_classes=self._preset_cfg["NUM_JOINTS"])

        self.softmax2d = nn.Softmax2d()

    def _initialize(self):
        for m in self.regressionModel.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classificationModel.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.preact(x)
        reg_weight = self.classificationModel(x)
        reg_weight = self.softmax2d(reg_weight)

        # channel last
        reg = self.regressionModel(x).permute(0, 2, 3, 1).contiguous()
        reg = self.sum_anchors(reg).permute(0, 3, 1, 2).contiguous()
        
        P_keys = reg_weight * reg

        return P_keys
