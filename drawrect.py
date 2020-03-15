#!/usr/bin/python
# -*- coding: utf8 -*-

from dataset.dataLoader import dataLoader_lung
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
from easydict import EasyDict as edict
from preprocessing.extrac_lung import get_segmented_lungs

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
    transform_list.append(transforms.CenterCrop((448, 448)))
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

def nms_reduce(top_index, top_pro, bound=8, size=56, filter_num=3):
    # 按照概率从小到大重排序,防止一个点有多个bbox导致小的值覆盖大的
    tmp = sorted(zip(top_pro, top_index))
    top_pro = [item[0] for item in tmp]
    top_index = [item[1] for item in tmp]

    fea_map = np.zeros(size * size, dtype=np.float32)
    fea_map[top_index] = top_pro
    #print('fea_map[top_index]', fea_map.shape, fea_map[top_index])

    # print(fea_map)
    # print(top_index, fea_map.sum())
    fea_map = fea_map.reshape((size, size))
    top_k_idx = (-fea_map).flatten().argsort()[0:10]
    #print(top_k_idx)
    top_k_idx = [item for item in top_k_idx if item in top_index]
    #print(top_k_idx)
    # print('top_index', top_index)
    for n in top_k_idx:
        # print('====', n, size, n%size)
        row, col = n // size, n % size
        # print('row, col',n, row, col)
        # print(pos)
        temp = fea_map[row, col]
        # print('tmp', temp)
        if temp > 0:
            # print("Bound 0")
            fea_map[max(row - bound, 0):min(row + bound, size), max(col - bound, 0):min(col + bound, size)] = 0
            fea_map[row, col] = temp
        else:
            pass
    # print(fea_map.sum())
    reduce_index = (-fea_map).flatten().argsort()
    large_then_zero = np.where(fea_map.flatten() > 0)[0]
    # print(large_then_zero)
    res = [item for item in reduce_index if (item in top_index and item in large_then_zero)][:3]
    res_pro = fea_map.flatten()[res]
    #print('top_index, top_pro', top_index, top_pro, res, res_pro)
    return list(zip(res, res_pro))


def draw_patch(epoch, model, index2classlist, args, dataroot, selected_ind):
    """Implement: use model to predict images and draw ten boxes by POOL6
    path to images need to predict is in './dataset/bird'

    result : directory to accept images with ten boxes
    subdirectory is epoch, e,g.0,1,2...

    index2classlist : transform predict label to specific classname
    """
    train_loader, valid_loader, test_loader = dataLoader_lung()
    
    for img_batch, targets, paths in test_loader:
        for img, target, path in zip(img_batch, targets, paths):

            #img = Image.open(path)
            # img = np.squeeze(img, axis=0)
            # img = Image.fromarray(img.numpy())

            #img_path = os.path.join(path_img, '{}.jpg'.format(original))

            #transform1 = get_transform()       # transform for predict
            #transform2 = transform_onlysize()
            #img = Image.open(img_path)
            #img_pad = transform2(img)
           # img_tensor = transform1(img)
            #img_tensor = data.unsqueeze(0)
            out1, out2, out3, x_p, indices = model(img.unsqueeze(0))
            x_p = x_p.squeeze().view(args.nclass, args.n_filters).cpu().detach().numpy()
            indices = indices.squeeze().view(args.nclass, args.n_filters).cpu().detach().numpy()
            #x_p, indices = x_p.reshape((args.nclass, args.n_filters)), indices.reshape(args.nclass, args.n_filters)
            out = out1 + out2 + 0.1 *out3
            #img = transform1(img)

            value, index = torch.max(out.cpu(), 1)
            vrange = np.arange(0, args.n_filters)
            # select from index - index+9 in 2000
            # in test I use 1st class, so I choose indices[0, 9]
            idx = int(index[0])
            img = Image.open(path)
            for indice, pro in nms_reduce(indices[0], x_p[0]):
                # 92为vgg指定成感受野的大小
                gsy = 92/2
                row, col = indice/56, indice%56
                p_tl = (8*col-gsy, 8*row-gsy)
                p_br = (col*8+gsy, row*8+gsy)
                img=img.convert('RGB')
                draw = ImageDraw.Draw(img)
                draw.rectangle((p_tl, p_br), outline='red',width=3)
                pro = str(round(pro, 2))
                font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=32)
                p_br = p_br[0] - 85, p_br[1] - 30
                draw.text(p_br, pro, font=font, fill='red')
            # search corresponding classname
            dirname = index2classlist[idx]
            input_file_name = os.path.basename(path)
            filename = 'epoch_'+'{:0>3}'.format(epoch)+'_[org]_'+str(target)+'_[predict]_'+str(idx)+str(dirname)+'_'+input_file_name
            result = os.path.abspath(args.result)
            tmp_path = f'{result}/{epoch:03}'
            os.makedirs(tmp_path, exist_ok=True)
            filepath = os.path.join(tmp_path,filename)
            img.save(filepath, "PNG")
        break

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

def predict_images(model, imgs, filters=5, output='./output/rep_img', threshold=0.6):
    import uuid
    import matplotlib.pyplot as plt
    images = []
    fold = f'{output}/{uuid.uuid4()}'
    os.makedirs(fold)

    max_predict = 0
    for sn, input_dict in enumerate(imgs):
        input_dict = edict(input_dict)
        # print(input_dict)
        img = input_dict.image
        img, _, per = get_segmented_lungs(img)
        if per <0.2 or per >0.5:
            continue
        lung_img = img.copy()
        file_name = os.path.basename((input_dict.filenane))
        out_file = f'{fold}/{file_name}.jpg'
        plt.imsave(out_file, img, cmap='gray')
        img = Image.fromarray(img)

        transform2 = transform_onlysize()

        out1, out2, out3, x_p, indices = model(transform2(img).unsqueeze(0).cuda())
        x_p = x_p.squeeze().view(2, filters).cpu().detach().numpy()
        indices = indices.squeeze().view(2, filters).cpu().detach().numpy()
        # x_p, indices = x_p.reshape((2,filters)), indices.reshape(2,filters)

        # print('x_p.shape', x_p.shape,indices.shape, x_p)
        # TODO replace with real prediction
        prediction = 0.88
        # print(out1, out2, out3, indices)

        out = out1 + out2 + 0.1 * out3
        # img = transform1(img)

        value, index = torch.max(out.cpu(), 1)
        # print(out.cpu())
        vrange = np.arange(0, filters)
        # select from index - index+9 in 2000
        # in test I use 1st class, so I choose indices[0, 9]
        idx = int(index[0])

        lung_img = cv2.resize(lung_img, (448,448))
        print(lung_img.shape)
        lung_img = Image.fromarray(lung_img)
        # 92为vgg指定成感受野的大小
        gsy = 92 / 2
        print(indices)
        # print('====', edict(dict(zip( indices[0], x_p[0],))))
        max_pro_sing_file = 0
        for indice, pro in nms_reduce(indices[0], x_p[0]):
            max_pro_sing_file = max(max_pro_sing_file, sigmoid(pro))
            # if sigmoid(pro) < threshold:
            #     continue
            row, col = indice / 56, indice % 56
            p_tl = (8 * col - gsy, 8 * row - gsy)
            p_br = (col * 8 + gsy, row * 8 + gsy)
            lung_img = lung_img.convert('RGB')

            draw = ImageDraw.Draw(lung_img)
            draw.rectangle((p_tl, p_br), outline='red', width=3)
            pro = str(round(pro, 2))
            font = ImageFont.truetype('./input/FreeMono.ttf', size=32)
            p_br = p_br[0] - 85, p_br[1] - 30
            draw.text(p_br, pro, font=font, fill='red')

        cur_pro = sigmoid(max_pro_sing_file)
        max_predict = max(max_predict, cur_pro)

        input_dict.pop('image')
        input_dict.prediction = cur_pro
        input_dict.path = out_file
        images.append(input_dict)
        lung_img.save(out_file)

    print(f'{len(images)} predict img save to :{fold}')
    return edict({'prediction': max_predict, 'images': images})

