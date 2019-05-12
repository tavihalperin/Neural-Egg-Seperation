import torch.nn as nn

class Generator64(nn.Module):
    def __init__(self, nc_in, nc_out, ndim):

        super(Generator64, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(nc_in, ndim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.l2 = nn.Sequential(
            nn.Conv2d(ndim, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.l3 = nn.Sequential(
            nn.Conv2d(ndim * 2, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.l4 = nn.Sequential(
            nn.Conv2d(ndim * 4, ndim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 8, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.ReLU(True))

        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 4, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.ReLU(True))

        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 2,     ndim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(True))

        self.l8 = nn.Sequential(
            nn.ConvTranspose2d(ndim, nc_out, 4, 2, 1, bias=True),
        )
        self.sig = nn.Sigmoid()


    def forward(self, input, use_sigmoid=True):
        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        if use_sigmoid:
            out = self.sig(out)
        return out




class Generator257(nn.Module):
    def __init__(self, nc_in, nc_out, ndim):

        super(Generator257, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(nc_in, ndim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.l2 = nn.Sequential(
            nn.Conv2d(ndim, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.LeakyReLU(0.2, inplace=True))

        self.l3 = nn.Sequential(
            nn.Conv2d(ndim * 2, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.l4 = nn.Sequential(
            nn.Conv2d(ndim * 4, ndim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 8),
            nn.LeakyReLU(0.2, inplace=True))

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 8, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.ReLU(True))

        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 4, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.ReLU(True))

        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(ndim * 2,     ndim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(True))

        self.l8 = nn.Sequential(
            nn.ConvTranspose2d(ndim, nc_out, 4, 2, 1, bias=True),
        )
        self.pad = nn.ReflectionPad2d((0,0,0,1))
        self.sig = nn.Sigmoid()


    def forward(self, input, use_sigmoid=True):
        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        out = self.pad(out)
        if use_sigmoid:
            out = self.sig(out)
        return out
