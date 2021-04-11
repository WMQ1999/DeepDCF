import torch
import numpy as np 
import torchvision 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.io.image import read_image
from torchvision.transforms import functional
from params import mean_rgb

def padding_frame(img, frame_sz, center_x, center_y, width, height):
    w, h = width/2, height/2
    padding_left = max(0, -int(np.round(center_x - w)))
    padding_top = max(0, -int(np.round(center_y - h)))
    padding_right = max(0, int(np.round(w + center_x  - frame_sz[-1])))
    padding_bottom = max(0, int(np.round(h + center_y - frame_sz[-2])))
    padding_img = torchvision.transforms.functional.pad(img, [padding_left, padding_top, padding_right, padding_bottom])
    return padding_img, padding_left, padding_top

def multiscales_cr(img, center_x, center_y, width, height, scales, re_sz):
    res_imgs = list()
    padding_img, x_offset, y_offset = padding_frame(img, img.shape, center_x, center_y, width*scales[-1], height*scales[-1])

    for scale in scales:
        sw, sh = scale * width, scale * height
        resized_img = functional.resized_crop(padding_img, int(np.ceil(center_y + y_offset - sh/2)), int(np.ceil(center_x + x_offset- sw/2)), int(sh), int(sw), (re_sz, re_sz))
        resized_img[0, :, :] -= mean_rgb[0]
        resized_img[1, :, :] -= mean_rgb[1]
        resized_img[2, :, :] -= mean_rgb[2]
        res_imgs.append(resized_img)

    res_imgs = torch.stack(res_imgs)
    return res_imgs

        
def cr(img, center_x, center_y, width, height, re_sz):

    padding_img, x_offset, y_offset = padding_frame(img, img.shape, center_x, center_y, width, height)
    resized_img =  functional.resized_crop(padding_img, int(np.ceil(center_y + y_offset - height/2)), int(np.ceil(center_x + x_offset -  width/2)), int(height), int(width), (re_sz, re_sz))
    resized_img[0, :, :] -= mean_rgb[0]
    resized_img[1, :, :] -= mean_rgb[1]
    resized_img[2, :, :] -= mean_rgb[2]

    return resized_img



def show_imgs(img, center_xs, center_ys, widths, heights):
    cnt = len(center_xs)
    fig, axs = plt.subplots(3, np.ceil(cnt / 3))
    mat_img = img / 255.0
    mat_img = mat_img.permute((1, 2, 0))

    for i, axe in enumerate(axs):
        if i == 0:
            axe.set_title('img')
        else:
            axe.set_title('conv%d'%i)

        rec = patches.Rectangle((center_xs[i], center_ys[i]), widths[i], heights[i], linewidth=2, edgecolor='r', fill=False) 
        axe.imshow(mat_img)
        axe.add_patch(rec)

    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()
    
def show_res(img, center_xs, center_ys, widths, heights):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    fig = plt.figure()
    axe = plt.subplot()
    mat_img = img / 255.0
    mat_img = mat_img.permute((1, 2, 0))
    axe.imshow(mat_img)

    for i,c in enumerate(colors):
        rec = patches.Rectangle((center_xs[i], center_ys[i]), widths[i], heights[i], linewidth=2, edgecolor=c , fill=False)
        if i == 0:
            rec.set_label('img')
        else:
            rec.set_label('conv%d'%i)

        axe.add_patch(rec)
    
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()



