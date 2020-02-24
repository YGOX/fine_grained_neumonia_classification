import h5py
import numpy as np
import nibabel
import glob
import pandas as pd
import re
import cv2
from configurations.paths import paths, file_names
import os
from numpy import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"

images_folder = paths['data']['preprocessed_data']
label_file = os.path.join(paths['data']['datainfo_path'], file_names['data']['patient_label'])

files = glob.glob(images_folder)
labels_df = pd.read_csv(label_file)

hdf5_path = os.path.join(paths['data']['datainfo_path'], file_names['data']['data_hdf5_file'])
images = []
ids= []
labels =[]
for idx, row in labels_df.iterrows():
    selected_idx = [indx for indx,f in enumerate(files) if row['id'] in files[indx]]
    if not selected_idx:
        continue
    else:
        label = labels_df.loc[labels_df['id'] == row['id'], 'cancer'].iloc[0]
        # print(label)
        img = np.load(files[selected_idx[0]])
        axial_slices= []
        PID=[]
        Plabel= []
        num_slices= 0
        #select axial slices in hthe middle, dump the first 50 and the last 50
        for i in np.arange(50, img.shape[1]-50,1):
            res = cv2.resize(img[0,:,i,:], dsize=(448, 448))
            axial_slices.append(np.expand_dims(res, axis=0))
            PID.append(row['id'])
            Plabel.append(label)
            num_slices+=1 
    images.append(np.concatenate(axial_slices))
    ids.append(np.repeat(row['id'], num_slices))
    labels.append(np.repeat(label, num_slices))

images= np.expand_dims(np.concatenate(images), axis=0)
images= moveaxis(images, 1, 0)
ids= np.concatenate(ids)
labels= np.concatenate(labels)

hdf5_file = h5py.File(hdf5_path, mode='w')

data_shape4 = images.shape
asciiList = [n.encode("ascii", "ignore") for n in ids.tolist()]
hdf5_file.create_dataset("ID", (len(asciiList),),dtype='S10')
hdf5_file.create_dataset("Image", images.shape, np.float32)
hdf5_file.create_dataset("Label", (labels.shape[0],), dtype=np.uint8)

hdf5_file["image"] = images
hdf5_file["label"] = labels
hdf5_file["iD"]= asciiList
hdf5_file.close()

