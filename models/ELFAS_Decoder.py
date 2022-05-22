import torch.nn as nn
import torch
import torch.nn.functional as F
import math

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

class UPDeconv(nn.Module):
    def __init__(self, up_sample_inp, de_invert_inp, oup, expand_ratio, inverted_n=2, se=True):
        super(UPDeconv, self).__init__()


        self.up_sample = nn.ConvTranspose2d(up_sample_inp, oup, kernel_size=2, stride=2, padding=0)
        layers = []
        layers.append(InvertedResidual(de_invert_inp, oup, stride=1, expand_ratio=expand_ratio, se=se))
        for i in range(inverted_n-1):
            layers.append(InvertedResidual(oup, oup, stride=1, expand_ratio=expand_ratio, se=se))

        self.de_invert = nn.Sequential(*layers)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(de_invert_inp, oup, kernel_size=1),
            nn.BatchNorm2d(oup)
        )

    def forward(self, encoder_feature, x):
        up_sample = self.up_sample(x)
        concat = torch.cat([up_sample, encoder_feature], dim=1)
        res_connect = self.conv1x1(concat)

        return res_connect + self.de_invert(concat)

class UPBilinear(nn.Module):
    def __init__(self, up_sample_inp, de_invert_inp, oup, expand_ratio, inverted_n=2, se=True):
        super(UPBilinear, self).__init__()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear'),
            nn.Conv2d(up_sample_inp, oup, kernel_size=1),
            nn.BatchNorm2d(oup),
        )

        layers = []
        layers.append(InvertedResidual(de_invert_inp, oup, stride=1, expand_ratio=expand_ratio, se=se))
        for i in range(inverted_n-1):
            layers.append(InvertedResidual(oup, oup, stride=1, expand_ratio=expand_ratio, se=se))

        self.de_invert = nn.Sequential(*layers)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(de_invert_inp, oup, kernel_size=1),
            nn.BatchNorm2d(oup)
        )

    def forward(self, encoder_feature, x):
        up_sample = self.up_sample(x)
        concat = torch.cat([up_sample, encoder_feature], dim=1)
        res_connect = self.conv1x1(concat)

        return res_connect + self.de_invert(concat)

# our global spatial attention mechanism
class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channel, hidden, size, equal_channel):
        super(GlobalSpatialAttention, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.hidden_size = hidden
        self.equal_channel = equal_channel

        self.horizontal = nn.LSTM(input_size=in_channel, hidden_size=self.hidden_size, batch_first=True,
                                bidirectional=True)  # each row
        self.vertical = nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True)  # each column

        self.conv_right = nn.Conv2d(self.hidden_size*2, self.equal_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_left = nn.Conv2d(self.in_channel, self.equal_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=2)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, input):

        batch, channel, height, width = input.size()
        x = torch.transpose(input, 1, 3)  # batch, width, height, in_channel

        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])  #
            temp.append(h)  # batch, width, 512
        horizontal_feature = torch.stack(temp, dim=1)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.vertical(horizontal_feature[:, :, i, :])
            temp.append(h)  # batch, width, 512
        vertical_feature = torch.stack(temp, dim=2)  # batch, height, 512, width

        vertical_feature = vertical_feature.permute(0, 3, 1, 2)
        vertical_feature = self.conv_right(vertical_feature)
        vertical_feature = vertical_feature.contiguous().view(batch, -1, height*width)
        vertical_feature = self.softmax(vertical_feature)

        q = self.conv_left(input)
        q_avg = self.avg_pool(q)

        q_avg = q_avg.contiguous().view(batch, 1, -1)
        q_avg = self.softmax(q_avg)

        attention = torch.matmul(q_avg, vertical_feature)
        attention = attention.view(batch, 1, height, width)
        attention = torch.sigmoid(attention)

        out = input * attention
        return out*self.alpha + input

class Decoder(nn.Module):
    def __init__(self, is_deconv=True, se=True):
        super(Decoder, self).__init__()

        self.se = se

        if is_deconv:
            UP = UPDeconv
            print('using deconv')
        else:
            UP = UPBilinear
            print('using bilinear')
        self.up1 = UP(up_sample_inp=96, de_invert_inp=64, oup=32, expand_ratio=6, inverted_n=2, se=self.se)
        self.up2 = UP(up_sample_inp=32, de_invert_inp=48, oup=24, expand_ratio=6, inverted_n=2, se=self.se)
        self.up3 = UP(up_sample_inp=24, de_invert_inp=32, oup=16, expand_ratio=6, inverted_n=2, se=self.se)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0),
            conv_3x3_bn(16, 3, stride=1)
        )
        self.global_att = GlobalSpatialAttention(hidden=128, size=14, in_channel=96, equal_channel=128)


    def forward(self, encoder_features):
        e0, e1, e2, e3, e4 = encoder_features

        # perform global pixel-wise attention to e4
        # e4 = self.global_att(e4)

        d1 = self.up1(e3, e4)
        d2 = self.up2(e2, d1)
        d3 = self.up3(e1, d2)
        spoof_noise = self.up4(d3)

        spoof_noise = torch.sigmoid(spoof_noise)

        return spoof_noise

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def decoder(is_deconv=True, se=False):
    model = Decoder(is_deconv, se)

    return model


if __name__ == '__main__':
    from models.ELFAS_Encoder import encoder
    encoder = encoder(pretrained=False, se=False)
    decoder = decoder(is_deconv=True, se=False)


    input = torch.randn((1, 3, 224, 224))
    encoder_output = encoder(input)
    spoof_noise = decoder(encoder_output)
    # print(e4.shape)
    print(spoof_noise.shape)
