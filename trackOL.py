import numpy as np 
import torch
import matplotlib.pyplot as plt
import os
import layers
from torchvision.io.image import read_image, ImageReadMode
from params import *
import image
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from torch.autograd.functional import jacobian, hessian
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_gt(gt_dir):
    gts = list()
    with open(gt_dir, 'r') as labels:
        for line in labels:
            temp = list()
            for num in line.split(','):
                temp.append(float(num))
            gts.append(temp)
    return gts

def create_label(label_shape):
    h, w = label_shape
    gaussian_func = MultivariateNormal(torch.tensor([h/2, w/2]), 2 * torch.eye(2))
    label = torch.zeros(h, w)
    
    for x in range(h):
        for y in range(w):
            label[x][y] = np.exp(gaussian_func.log_prob(torch.tensor([x, y])))
    
    return label


def subgrid_detect(raw_score, context_width, context_height, x, y):
    M, N = raw_score.shape
    def func(input):
        out = torch.zeros(1, dtype = torch.complex64)
        for m in range(M):
            for n in range(N):
                out += raw_score[m][n] * torch.exp(1j * math.pi * 2 * (m/M * input[0] + n/N*input[1]))

        return out
    current = torch.tensor([x, y])
    while(True):
        jaco = jacobian(func, current).to(dtype = torch.float32)
        hess = hessian(func, current).to(dtype = torch.float32)
        inverse_hess = torch.inverse(hess)
        last = torch.clone(current)
        current = last - torch.squeeze(torch.matmul(inverse_hess, jaco.transpose(0, 1)))
        if(torch.dist(current, last) < 1e2):
            break

    dis_y, dis_x = torch.tensor([M/2, N/2]) - current
    dis_y = dis_y * context_height / M
    dis_x = dis_x * context_width / N
    return dis_x, dis_y

vgg_m = layers.VGG_M()
vgg_m.load_state_dict(torch.load('./States/vgg_param.pth'))
scales =  scale_factor ** np.linspace(np.floor((1 -scale_num)/2), np.floor((scale_num - 1)/2), num = scale_num)



for video in track_videos:
    imgs_dir = os.path.join(dataset_dir, video)
    gt_dir = os.path.join(imgs_dir, gt_file)
    img_dir = os.path.join(imgs_dir, f'{1:08d}.png')
    gts = get_gt(gt_dir)
    cnt = len(gts)

    pos_x, pos_y, width, height = gts[0]
    center_x, center_y = width/2, height/2
    context_width, context_height = context_factor * width, context_factor*height

    img = read_image(img_dir, ImageReadMode.RGB).to(dtype = torch.float32)
    cr_img = image.cr(img, center_x, center_y, context_width, context_height, re_sz)
    cr_img = torch.unsqueeze(cr_img, dim = 0)
    conv1_img, conv2_img, conv3_img, conv4_img, conv5_img = vgg_m(cr_img)
    
    x_DCF = layers.DCF(cr_img, create_label(cr_img.shape[-2:]))
    conv1_DCF = layers.DCF(conv1_img, create_label(conv1_img.shape[-2:]))
    conv2_DCF = layers.DCF(conv2_img, create_label(conv2_img.shape[-2:]))
    conv3_DCF = layers.DCF(conv3_img, create_label(conv3_img.shape[-2:]))
    conv4_DCF = layers.DCF(conv4_img, create_label(conv4_img.shape[-2:]))
    conv5_DCF = layers.DCF(conv5_img, create_label(conv5_img.shape[-2:]))
    DCFS = [x_DCF, conv1_DCF, conv2_DCF, conv3_DCF, conv4_DCF, conv5_DCF]
    convs = [None, vgg_m.conv1, vgg_m.conv2, vgg_m.conv3, vgg_m.conv4, vgg_m.conv5]
    widths = [[width for i in range(6)]]
    heights = [[height for i in range(6)]]
    context_heights = [[context_height for i in range(6)]]
    context_widths = [[context_width for i in range(6)]]
    center_xs = [[center_x for i in range(6)]]
    center_ys = [[center_y for i in range(6)]]

    for i in range(1, cnt):
        img_dir = os.path.join(imgs_dir, f'{i+1:08d}.png')
        img = read_image(img_dir, ImageReadMode.RGB).to(dtype = torch.float32)
        widths.append([])
        heights.append([])
        center_xs.append([])
        center_ys.append([])
        context_heights.append([])
        context_widths.append([])
        for j in range(6):
            center_x, center_y, width, height = center_xs[-2][j], center_ys[-2][j], widths[-2][j], heights[-2][j]
            context_height, context_width = context_heights[-2][j], context_widths[-2][j]
            cr_imgs = image.multiscales_cr(img, center_x, center_y, context_width,  context_height, scales, re_sz)
            
            for k in range(j+1):
                if convs[k]:
                    cr_imgs  = convs[k](cr_imgs)
            
            raw_score, scale_index, x, y = DCFS[j].forward(cr_imgs)
            dis_x, dis_y = subgrid_detect(raw_score[scale_index], context_width, context_height, x, y)
            center_x += dis_x*scales[scale_index]
            center_y += dis_y*scales[scale_index]
            width *= scales[scale_index]
            height *= scales[scale_index]
            context_height *= scales[scale_index]
            context_width *= scales[scale_index]
            center_xs[-1].append(center_x)
            center_ys[-1].append(center_y)
            widths[-1].append(width)
            heights[-1].append(height)
            context_heights[-1].append(context_height)
            context_widths[-1].append(context_width)

            cr_img = image.cr(img, center_x, center_y, context_width, context_height, re_sz)
            cr_img = torch.unsqueeze(cr_img, dim = 0)

            for k in range(j + 1):
                if convs[k]:
                    cr_img = convs[k](cr_img)
            DCFS[i].update(cr_img)
            
        image.show_imgs(img, center_x[-1], center_y[-1], widths[-1], heights[-1])
