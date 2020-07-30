import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

        # CHECK INITIAL CHANNELS HERE
        self.conv1 = downStep(1, 64)
        self.pool1 = poolLayer()
        self.conv2 = downStep(64, 128)
        self.pool2 = poolLayer()
        self.conv3 = downStep(128, 256)
        self.pool3 = poolLayer()
        self.conv4 = downStep(256, 512)
        self.pool4 = poolLayer()
        self.conv5 = downStep(512, 1024)

        self.conv6 = upStep(1024, 512)
        self.conv7 = upStep(512, 256)
        self.conv8 = upStep(256, 128)
        self.conv9 = upStep(128, 64, False)

        self.conv10 = plainConv(64, 2)

    def forward(self, x):
        # todo
        x1 = self.conv1(x)
        x = self.pool1(x1)

        #x1 = crop(x1, 88, 568)
        x1 = x1[:, :, 88:480, 88:480]
        #print('x1')
        #print(x1.size())

        x2 = self.conv2(x)
        x = self.pool2(x2)
        #x2 = crop(x2, 40, 280)
        x2 = x2[:, :, 40:240, 40:240]
        #print('x2')
        #print(x2.size())

        x3 = self.conv3(x)
        x = self.pool3(x3)
        #x3 = crop(x3, 16, 135)
        x3 = x3[:, :, 16:120, 16:120]
        #print('x3')
        #print(x3.size())

        x4 = self.conv4(x)
        x = self.pool4(x4)
        #x4 = crop(x4, 4, 64)
        x4 = x4[:, :, 4:60, 4:60]
        #print('x4')
        #print(x4.size())

        x = self.conv5(x)

        x = self.conv6(x, x4)
        x = self.conv7(x, x3)
        x = self.conv8(x, x2)
        x = self.conv9(x, x1)
        x = self.conv10(x)

        return x

    def crop(x, a, b):
        return x[a:b - a, a:b - a, :]


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.conv = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=3),
                                  nn.ReLU(),
                                  nn.Conv2d(outC, outC, kernel_size=3),
                                  nn.ReLU()
                                  )

        # self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # todo
        x = self.conv(x)
        # if(poolCheck == 1):
        # x = self.pool(x)
        return x


class poolLayer(nn.Module):
    def __init__(self):
        super(poolLayer, self).__init__()
        # todo

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # todo

        x = self.pool(x)
        return x


class cropMap(nn.Module):
    def __init__(self):
        super(cropMap, self).__init__()
        # todo
        self.conv = nn.crop2

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # todo
        x = self.conv(x)
        if (poolCheck == 1):
            x = self.pool(x)
        return x


class plainConv(nn.Module):
    def __init__(self, inC, outC):
        super(plainConv, self).__init__()

        self.conv = nn.Conv2d(inC, outC, kernel_size=1)

    def forward(self, x):
        # todo
        x = self.conv(x)

        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo

        # transpose upconvolution
        self.upscale = nn.Sequential(nn.ConvTranspose2d(inC, outC, kernel_size=2, stride=2))

        if (withReLU == True):
            self.conv = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=3),
                                      nn.ReLU(),
                                      nn.Conv2d(outC, outC, kernel_size=3),
                                      nn.ReLU())

        if (withReLU == False):
            self.conv = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=3),
                                      nn.Conv2d(outC, outC, kernel_size=3),
                                      )

        # Do not forget to concatenate with respective step in contracting path - to implemengt skip connections

        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons! - the last 2 basically

    def forward(self, x, x_down):
        # todo
        x = self.upscale(x)
        #print(x.size())
        #print(x_down.size())
        x = torch.cat([x_down, x], dim=1)
        x = self.conv(x)

        return x
