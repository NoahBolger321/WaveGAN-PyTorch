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


class PhaseShuffle(nn.Module):
    def __init__(self, n):
        super(PhaseShuffle, self).__init__()
        self.n = n

    def forward(self, x):
        b, x_len, nch = list(x.size())
        phase = torch.FloatTensor(1, x_len).uniform_(-self.n, self.n+1)
        pad_left = torch.maximum(phase, 0)
        pad_right = torch.maximum(-phase, 0)
        phase_start = pad_right
        x = nn.functional.pad(x, ((0, 0, pad_left, pad_right, 0, 0)), mode='reflect')
        x = x[:, phase_start:phase_start+x_len]
        return torch.reshape(x, (b, x_len, nch))


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


class Discriminator(nn.Module):
    def __init__(self, num_gpu):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(25, nc, d, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2.0),
            nn.ConvTranspose1d(25, d, 2*d, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2.0),
            nn.ConvTranspose1d(25, 2*d, 4*d, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2.0),
            nn.ConvTranspose1d(25, 4*d, 8*d, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(2.0),
            nn.ConvTranspose1d(25, 8*d, 16*d, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape((nz, 16*16*d)),
            nn.Linear(4 * 4 * 16 * d, 1),
        )


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

    netD = Discriminator(1).to(device)
    netD.apply(weights_init)
    print(netD)
