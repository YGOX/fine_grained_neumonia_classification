import numpy as np
num_classes=2 
latent_dim = 512
name_classes = np.asarray(['Negative', 'Positive'])
num_conv = 5
	
#img_shape.astype(np.int32)
bl_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels': 32,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv2': {
		'in_channels': 32,
		'out_channels': 32,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv3': {
		'in_channels': 32,
		'out_channels': 64,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv4': {
		'in_channels': 64,
		'out_channels': 64,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},

	'conv5': {
		'in_channels': 64,
		'out_channels': 128,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},
	'conv6': {
		'in_channels': 128,
		'out_channels': 128,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},

	'conv7': {
		'in_channels': 128,
		'out_channels': 256,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},
	'conv8': {
		'in_channels': 256,
		'out_channels': 256,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},
	'conv9': {
		'in_channels': 256,
		'out_channels': 512,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},
	'conv10': {
		'in_channels': 512,
		'out_channels': 512,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1},
	'conv11': {
		'in_channels': 512,
		'out_channels': num_classes,
		'kernel_size': 1,
		'stride': 1,
		'padding': 0},

	'maxpool2d': {

			'kernel': 2,
			'stride': 2

	}
}


params = {
	'model': {
		'conv_drop_prob': 0.4,
		'fcc_drop_prob'	: 0.0
	},
	
	'train'	:	{
		'model'				: 'BL',
		'seed'				: 42,
		'learning_rate' 	: 0.0001,
		'num_epochs' 		: 100,
		'batch_size' 		: 28,
		'lr_schedule'		: [35]
	}
}
