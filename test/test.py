from drawrect import predict_images
import matplotlib.pyplot as plt

import pydicom
from glob import glob
from PIL import Image, ImageFont, ImageDraw
from drawrect import *
from easydict import EasyDict as edict


imgs = []
for file in glob('./input/sample/*.*'):
    # print(file)
    ds = pydicom.dcmread(file, force=True)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    img = ds.pixel_array
    # print(img.shape)

    imgs.append({'index':ds.SeriesNumber, 'image':img, 'filenane':file })

from model.DFL import *

filters=3
model = DFL_VGG16(k=filters, nclass=2).cuda()

res = predict_images(model, imgs, filters=filters)

print(res)