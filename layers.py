import torch
import torchvision
from torch import nn
from torchsummary import summary
from params import *
import numpy as np

class VGG_M(nn.Module):
    def __init__(self):
        super(VGG_M, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 7, 2),
            nn.LocalResponseNorm(k = 2, size = 5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 2, 1),
            nn.LocalResponseNorm(k = 2, size = 5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        return out1, out2, out3, out4, out5
        #return out5


class DCF:
    def __init__(self, first_img, label):
        first_img = torch.squeeze(first_img, dim = 0)
        self.channel, self.height, self.width = first_img.shape[-3:]
        
        h_hann =  torch.hann_window(self.height)
        h_hann = torch.unsqueeze(h_hann, dim = 0)
        h_hann = torch.transpose(h_hann, -1, -2)
        w_hann = torch.hann_window(self.width)
        w_hann = torch.unsqueeze(w_hann, dim = 0)
        self.hann_window = torch.matmul(h_hann, w_hann)
        first_img = first_img * self.hann_window

        self.fourier_lable = torch.fft.fft2(label)
        self.fourier_img = torch.fft.fft2(first_img)
        self.numerator = torch.conj(self.fourier_lable) * self.fourier_img
        self.denominator = torch.sum(torch.conj(self.fourier_img) * self.fourier_img, dim = 0) + lambda_
    
    def forward(self, x):
        x = x * self.hann_window
        fourier_imgs = torch.fft.fft2(x)
        score = self.numerator / self.denominator * fourier_imgs
        raw_score = torch.sum(score, dim = -3)
        score = torch.fft.ifft2(raw_score).to(dtype = torch.float32)
        scale_index = torch.argmax(torch.amax(score, dim  = (-2, -1)))
        raw_pos = torch.argmax(score[scale_index]).numpy()
        pos_y = np.ceil(raw_pos / self.height)
        pos_x = raw_pos % self.height
        
        return raw_score, scale_index, pos_x, pos_y
    
    def update(self, x):
        x = torch.squeeze(x, dim = 0)
        x = x * self.hann_window
        self.fourier_img = torch.fft.fft2(x)
        self.numerator = (1. - _gamma) * self.numerator + _gamma * torch.conj(self.fourier_lable) * self.fourier_img
        self.denominator = (1. - _gamma) * self.denominator + _gamma * (torch.sum(torch.conj(self.fourier_img) * self.fourier_img, dim = -3) + lambda_)
        

# model = VGG_M().to('cuda')
# summary(model, (3, 224, 224))
