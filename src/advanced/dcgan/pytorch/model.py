import torch

class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_channels = args["n_channels"]
        self.h_dim = args["h_dim"]

        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(n_channels, h_dim, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (h_dim) x 32 x 32
            torch.nn.Conv2d(h_dim, h_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (h_dim*2) x 16 x 16
            torch.nn.Conv2d(h_dim * 2, h_dim * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (h_dim*4) x 8 x 8
            torch.nn.Conv2d(h_dim * 4, h_dim * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (h_dim*8) x 4 x 4
            torch.nn.Conv2d(h_dim * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.h_dim = args["h_dim"]
        self.Z_dim = args["Z_dim"]

        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(Z_dim, h_dim * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(h_dim * 8),
            torch.nn.ReLU(True),
            # state size. (h_dim*8) x 4 x 4
            torch.nn.ConvTranspose2d(h_dim * 8, h_dim * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim * 4),
            torch.nn.ReLU(True),
            # state size. (h_dim*4) x 8 x 8
            torch.nn.ConvTranspose2d(h_dim * 4, h_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim * 2),
            torch.nn.ReLU(True),
            # state size. (h_dim*2) x 16 x 16
            torch.nn.ConvTranspose2d(h_dim * 2, h_dim, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(h_dim),
            torch.nn.ReLU(True),
            # state size. (h_dim) x 32 x 32
            torch.nn.ConvTranspose2d(h_dim, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

