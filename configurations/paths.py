'''
contains paths used in the project
'''

import time
		
paths = {
	'data' : {
		'datapath': 	'/home/scat5837/baidunetdiskdownload/COVID19',
		'preprocess_result_path':'./prep_result1/',
		'preprocessed_data':'/home/scat5837/PycharmProjects/DFL-CNN/prep_result/*.npy',
		'preprocessed_data_label':'./prep_label/*label.npy',
		'datainfo_path':'/home/scat5837/Documents/Fine-grained Neumonia Classification/data_info'},
	'output'	:	{		
		'bl_base_folder'							:	'./bl_outputs/'
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
