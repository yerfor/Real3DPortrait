import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import numpy as np


class VGG19Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19Model()

        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
    
    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.y.downsample(y)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    
class VGG19Model(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(VGG19Model, self).__init__()
        # load pretrained weights from torchvision
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, is_normalize=True):
        if is_normalize: # x is within [-1,1]
            X = (X + 1) / 2
            assert torch.all(X <= 1+1e-6)
            assert torch.all(X >= -1e-6)
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

if __name__ == '__main__':
    vgg_loss_fn = VGG19Loss()
    x1 = torch.randn([4, 3, 512, 512]).clamp(-1,1)
    x2 = torch.randn([4, 3, 512, 512]).clamp(-1,1)
    loss = vgg_loss_fn(x1, x2)
    print(loss)