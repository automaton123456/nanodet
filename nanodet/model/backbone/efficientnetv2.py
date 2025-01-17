"""
Adapted from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
"""

import torch
import torch.nn as nn
import math

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                nn.ReLU6(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, model_name, out_stages=(1, 3, 5), activation="ReLU6", pretrain=True):
        width_mult=1
        cfgs = [
                # t, c, n, s, SE
                [1,  24,  2, 1, 0],
                [4,  48,  4, 2, 0],
                [4,  64,  4, 2, 0],
                [4, 128,  6, 2, 1],
                [6, 160,  9, 1, 1],
                [6, 256, 15, 2, 1],
        ]

        cfgs = [
                # t, c, n, s, SE
                [1,  24,  2, 1, 0],
                [4,  48,  2, 2, 0],
                [4,  64,  3, 2, 0],
                [4, 128,  3, 1, 1],
                [6, 160,  4, 2, 1],
                [6, 256, 1, 1, 1], #T is hidden convulutions
        ]
        
        v1_b0_block_str = [
            'r2_k3_s2_e6_i16_o24_se0.25',
            'r2_k5_s2_e6_i24_o40_se0.25',
            'r3_k3_s2_e6_i40_o80_se0.25',
            'r3_k5_s1_e6_i80_o112_se0.25',
            'r4_k5_s2_e6_i112_o192_se0.25',
            'r1_k3_s1_e6_i192_o320_se0.25',
        ]

        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        
        self.out_stages=out_stages
        self.activation="ReLU6"

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        self.stem = nn.Sequential(*layers)
        self.blocks = nn.ModuleList([])

        # building inverted residual blocks
        block = MBConv

        for t, c, n, s, use_se in self.cfgs:
            stage = nn.ModuleList([])

            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stage.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel

            self.blocks.append(stage) 

        
        # building last several layers
        #output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    

    def forward(self, x):
        output = []
        x = self.stem(x)
        counter = 0

        for stage in self.blocks:
          for block in stage:
            x = block(x)

          if counter in self.out_stages:
            output.append(x)  

          counter = counter + 1  
        
        
        #x = self.conv(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)

        #print("Oiii" + str(len(output)))
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    
    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
