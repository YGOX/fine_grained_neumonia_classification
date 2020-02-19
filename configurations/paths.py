'''
contains paths used in the project
'''

import time
		
paths = {
	'data' : {
		'datapath': 	'/home/scat5837/baidunetdiskdownload/stage1',
		'preprocess_result_path':'/home/scat5837/PycharmProjects/wuhan/prep_result/',
		'preprocessed_data':'/home/scat5837/PycharmProjects/DFL-CNN/prep_result/*.npy',
		'datainfo_path':'/home/scat5837/PycharmProjects/DFL-CNN/data_info/'},
	'output'	:	{		
		'bl_base_folder'							:	'/home/scat5837/PycharmProjects/wuhan/bl_outputs/'
	}
}

file_names = {
	'data'	:	{
		'data_hdf5_file'							: 	'data.hdf5',
		'patient_label'							: 	'stage1_labels.csv',
		'IDtodict'								:	'IDtodict.pkl',
		'Train_data_indices'						: 	'train_data_indices.pkl',
		'Valid_data_indices'						: 	'valid_data_indices.pkl',
		'Test_data_indices'						: 	'test_data_indices.pkl'
	},
	'output'	:	{
		'parameters'							:	'parameters.json',
		'train_loss_classification'				:	'train_loss_classification.pkl',
		'valid_loss'							:	'valid_loss.pkl',
		'train_accuracy'						:	'train_accuracy.pkl',
		'valid_accuracy'						:	'valid_accuracy.pkl',
		'train_f1_score'						:	'train_f1_score.pkl',
		'valid_f1_score'						:	'valid_f1_score.pkl',
		'best_val'						:	'valid_best.txt',
		'test_results'						:	'test_results.txt',
		'test_scores'							: 	'test_scores.hdf5'
	}
}
