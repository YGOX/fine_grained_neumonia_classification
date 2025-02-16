'''
pyTorch custom dataloader
'''
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from configurations.paths import paths, file_names
import os
import pickle
from random import shuffle
import pandas as pd
from dataset.splitDataset import getIndicesTrainValidTest
import torchvision.transforms as transforms
#import torchvision.transforms as transforms

class HDF5loader():
	def __init__(self, filename, trans=None, train_indices=None):
		f = h5py.File(filename, 'r',  libver='latest')
		self.img_f = f['image']
		self.trans = trans
		self.train_indices = train_indices
		self.labels = f['label']
		#calculate class frequency and inverse
		_, cls_count= np.unique(np.array(self.labels)[np.array(train_indices)], return_counts=True)
		cls_weights= 1/cls_count
		self.cls_weights= (cls_weights/cls_weights.sum()).tolist()

	def __getitem__(self, index):
		img = self.img_f[index]
		img= np.squeeze(img, axis=0)
		img = Image.fromarray(img)
		label = self.labels[index]
		# random transformation
		if self.trans is not None and index in self.train_indices:
			transform= transforms.Compose([
			transforms.Grayscale(num_output_channels=3),
			transforms.transforms.RandomHorizontalFlip(p=0.3),
			#transforms.RandomResizedCrop((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
			img= transform(img)
		else:
			transform= transforms.Compose([transforms.Grayscale(num_output_channels=3),
										   #transforms.CenterCrop((224, 224)),
										   transforms.ToTensor(),
										   transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
			img=transform(img)
		#img= img.numpy()

		#if np.std(img) != 0:  # to account for black images
		#	mean = np.mean(img)
		#	std = np.std(img)
		#	img = 1.0 * (img - mean) / std
		
		#img = img.astype(float)
		#img = torch.from_numpy(img).float()
		
		label = torch.LongTensor([label])
		
		return (img ,label.squeeze(-1))
		
	def __len__(self):
		return self.img_f.shape[0]

	def get_cls_weights(self):
		return self.cls_weights


def dataLoader(hdf5_file, train_indices, valid_indices, test_indices, num_workers, batch_size, trans=None):

	pin_memory = True

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(valid_indices)
	test_sampler = SubsetRandomSampler(test_indices)

	data = HDF5loader(hdf5_file, trans, train_indices=train_indices)
	cls_weights = data.get_cls_weights()

	train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler,
							  num_workers=num_workers, pin_memory=pin_memory)
	valid_loader = DataLoader(data, batch_size=1, sampler=valid_sampler,
							  num_workers=num_workers, pin_memory=pin_memory)
	test_loader = DataLoader(data, batch_size=1, sampler=test_sampler,
							 num_workers=num_workers, pin_memory=pin_memory)
	
	return (train_loader, valid_loader, test_loader, cls_weights)

