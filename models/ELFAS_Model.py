from models.ELFAS_Encoder import encoder
from models.ELFAS_Decoder import decoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class ELFAS(nn.Module):
    def __init__(self, pretrained=True, se=False, is_deconv=True):
        super(ELFAS, self).__init__()

        self.encoder = encoder(pretrained=pretrained, se=se)
        self.decoder = decoder(is_deconv=is_deconv, se=se)

    def forward(self, x):

        encoder_output = self.encoder(x)
        spoof_noise = self.decoder(encoder_output)

        return spoof_noise


def elfas_model():
    return ELFAS(pretrained=True, se=False, is_deconv=True)


if __name__ == '__main__':

    input = torch.randn((8, 3, 224, 224))
    model = elfas_model()

    spoof_noise = model(input)
    print('spoof_noise.shape', spoof_noise.shape)
