import torch
import torch.nn as nn


# number of channels
nc = 1

# batch size
b = 64

# model dimensionality
d = 64

# phase shuffle (WaveGAN)
ps = 2

# size of the input latent vector z
nz = 100

# filter length
fl = 25


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class Generator(nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            nn.Linear(nz, 4*4*16*d),
            Reshape((b, 16, d*16)),
            nn.ReLU(True),
            nn.ConvTranspose1d(25, 16*d, 8*d, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(25, 8*d, 4*d, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(25, 4*d, 2*d, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(25, 2*d, d, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(25, d, nc, stride=4),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    device = 'cpu'
    netG = Generator(1).to(device)
    netG.apply(weights_init)
    print(netG)
