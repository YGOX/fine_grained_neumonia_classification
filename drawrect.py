from model.DFL import DFL_VGG16
from utils.util import *
from utils.transform import *
from train import *
from validate import *
from utils.init import *
import sys
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import h5py
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from random import shuffle
import torchvision.datasets as datasets
#from utils.MyImageFolderWithPaths import ImageFolderWithPaths
from PIL import Image, ImageFont, ImageDraw
import os
import re
import numpy as np
import cv2
def scale_width(img, target_width):
    ow, oh = img.size
    w = target_width
    target_height = int(target_width * oh / ow)
    h = target_height
    return img.resize((w, h), Image.BICUBIC)
    
    
def transform_onlysize():
    transform_list = []
    #transform_list.append(transforms.Resize(224))
    transform_list.append(transforms.Grayscale(num_output_channels=3))
    #transform_list.append(transforms.CenterCrop((224, 224)))
    #transform_list.append(transforms.Pad((42, 42)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    return transforms.Compose(transform_list)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def read_specific_line(line, path):
    target = int(line)
    with open(path, 'r') as f:
        line = f.readline()
        c = []
        while line:
            currentline = line
            c.append(currentline)
            line = f.readline()
        
    reg =  c[target-1].split(',')[-1]     
    return reg

def path_to_contents(path):
    filename = path.split('/')[-1]
    index_gtline = re.split('_|.jpg', filename)[-2]
    index_image = filename.split('_')[1]
    gt_dir = '/data1/data_sdj/ICDAR2015/end2end/train/gt'
    gt_file = os.path.join(gt_dir, 'gt_img_'+str(index_image)+'.txt')
    # I want to read gt_file of specific line index_gtline
    contents = read_specific_line(int(index_gtline), gt_file)
    #print(index_image, index_gtline, contents)
    return contents

def create_font(fontfile, contents):
    # text and font
    unicode_text = contents
    if isinstance(unicode_text,str) and unicode_text.find('###') != -1 or unicode_text == '':
        print('######################')
        return None
    try:
        font = ImageFont.truetype(fontfile, 36, encoding = 'unic')
    
        # get line size
        # text_width, text_font.getsize(unicode_text)
    
        canvas = Image.new('RGB', (128, 48), "white")
    
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5), unicode_text, 'black', font)

    #canvas.save('unicode-text.png','PNG')
    #canvas.show()
        print(canvas.size)
        return canvas
    except:
        return None

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    #imga = Image.fromarray(imga)
    #imgb = Image.fromarray(imgb)
    w1,h1 = imga.size
    w2,h2 = imgb.size
    img = Image.new("RGB",(256, 48))
    img.paste(imga, (0,0))
    img.paste(imgb, (128, 0))
    return img

def get_transform():
    transform_list = []
    
    #transform_list.append(transforms.Lambda(lambda img:scale_keep_ar_min_fixed(img, 448)))

    #transform_list.append(transforms.ToPILImage())
    #transform_list.append(transforms.Grayscale())
    
    #transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
    
    transform_list.append(transforms.CenterCrop((224, 224)))
    
    #transform_list.append(transforms.ToTensor())
    
    #transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    
    return transforms.Compose(transform_list)


def draw_patch(epoch, model, index2classlist, args, dataroot, selected_ind):
    """Implement: use model to predict images and draw ten boxes by POOL6
    path to images need to predict is in './dataset/bird'

    result : directory to accept images with ten boxes
    subdirectory is epoch, e,g.0,1,2...

    index2classlist : transform predict label to specific classname
    """
    result = os.path.abspath(args.result)
    if not os.path.isdir(result):
        os.mkdir(result)
    model.eval()
    f = h5py.File(dataroot, 'r', libver='latest')
    img_f = f['image']
    imgs= img_f[selected_ind]
   #path_img = os.path.join(os.path.abspath('./'), 'vis_img')
    #num_imgs = len(os.listdir(path_img))

    dirs = os.path.join(result, str(epoch))
    if not os.path.exists(dirs):
        os.mkdir(dirs)
    
    for ind in np.arange(100):
        img = imgs[ind]
        img = np.squeeze(img, axis=0)
        img = Image.fromarray(img)

        #img_path = os.path.join(path_img, '{}.jpg'.format(original))
        
        #transform1 = get_transform()       # transform for predict
        transform2 = transform_onlysize()
        #img = Image.open(img_path)
        #img_pad = transform2(img)
       # img_tensor = transform1(img)
        #img_tensor = data.unsqueeze(0)
        #out1, out2, out3, indices = model(transform2(img).unsqueeze(0))
        out1, out2, indices = model(transform2(img).unsqueeze(0))
        #out = out1 + out2 + 0.1 *out3
        out = out1 + out2
        #img = transform1(img)
    
        value, index = torch.max(out.cpu(), 1)
        vrange = np.arange(0, args.n_filters)
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9]
        idx = int(index[0])
        for i in vrange:
            indice = indices[0, i]
            row, col = indice/56, indice%56
            p_tl = (8*col, 8*row)
            p_br = (col*8+92, row*8+92)
            img=img.convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.rectangle((p_tl, p_br), outline='red')
    
        # search corresponding classname
        dirname = index2classlist[idx]
        filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(ind)+'_[predict]_'+str(dirname)
        filepath = os.path.join(os.path.join(result,str(epoch)),filename)
        img.save(filepath, "PNG")

    
if __name__ == '__main__':
    draw_patch()
