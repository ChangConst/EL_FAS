import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# path to the pretrained mobilenetv2 model
# alter this to your location
mobilenetv2_model_path = '/home/chang/FAS_Project/preTrainedModels/mobileNet/mobilenet_v2-b0353104.pth'

# Inverted residual block from MobileNetV2

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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        # in_channels out_channels kernel_size stride padding bias
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, se):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)

        self.identity = stride == 1 and inp == oup

        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        if expand_ratio == 1:

            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # se layer if specified
                SELayer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # se layer if specified
                SELayer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, width_mult=1., se=False):
        super(Encoder, self).__init__()

        self.se = se

        E1_settings = [
            # t, c, n, s, se
            [1, 16, 1, 1, self.se]
        ]

        E2_settings = [
            # t, c, n, s, se
            [6, 24, 2, 2, self.se]
        ]

        E3_settings = [
            [6, 32, 3, 2, self.se]
        ]

        E4_settings = [
            [6, 64, 4, 2, self.se],
            [6, 96, 3, 1, self.se],
        ]

        block = InvertedResidual
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        self.E1 = [conv_3x3_bn(3, input_channel, 2)]
        for t, c, n, s, se in E1_settings:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                self.E1.append(block(input_channel, output_channel, s if i == 0 else 1, t, se))
                input_channel = output_channel
        self.E1 = nn.Sequential(*self.E1)

        self.E2 = []
        for t, c, n, s, se in E2_settings:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                self.E2.append(block(input_channel, output_channel, s if i == 0 else 1, t, se))
                input_channel = output_channel
        self.E2 = nn.Sequential(*self.E2)

        self.E3 = []
        for t, c, n, s, se in E3_settings:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                self.E3.append(block(input_channel, output_channel, s if i == 0 else 1, t, se))
                input_channel = output_channel
        self.E3 = nn.Sequential(*self.E3)

        self.E4 = []
        for t, c, n, s, se in E4_settings:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                self.E4.append(block(input_channel, output_channel, s if i == 0 else 1, t, se))
                input_channel = output_channel
        self.E4 = nn.Sequential(*self.E4)

    def forward(self, x):
        x1 = self.E1(x)
        x2 = self.E2(x1)
        x3 = self.E3(x2)
        x4 = self.E4(x3)

        return x, x1, x2, x3, x4

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def encoder(pretrained=True, se=False):

    model = Encoder(se=se)

    if pretrained:
        # apply parameters in the pretrained mobilenetv2 to initialize our encoder

        pre_parameters = torch.load(mobilenetv2_model_path)
        pre_parameters_keys = []
        for key in pre_parameters.keys():
            if not 'classifier' in key and int(key.split('.')[1]) < 14 and 'feature' in key:
                pre_parameters_keys.append(key)

        load_model_keys = []
        for key in model.state_dict().keys():
            if not 'fc' in key:
                load_model_keys.append(key)

        load_dict = {}
        for new_key, old_key in zip(load_model_keys, pre_parameters_keys):
            load_dict[new_key] = pre_parameters[old_key]

        model.load_state_dict(load_dict, strict=False)
        print(f"load: {mobilenetv2_model_path} success!")

    return model

if __name__ == '__main__':
    model = encoder(pretrained=False, se=False)

    input = torch.randn((8, 3, 224, 224))
    output = model(input)
    print('x.shape', output[0].shape)
    print('x1.shape', output[1].shape)
    print('x2.shape', output[2].shape)
    print('x3.shape', output[3].shape)
    print('x4.shape', output[4].shape)
