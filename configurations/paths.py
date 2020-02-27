'''
contains paths used in the project
'''

import time
		
paths = {
	'data' : {
		'datapath': 	'./input/stage1',
		'preprocess_result_path':'./output/prep_result/',
		'preprocessed_data':'./output/prep_result/*.npy',
		'preprocessed_data_label':'./output/prep_label/*label.npy',
		'datainfo_path':'./data_info/'},
	'output'	:	{		
		'bl_base_folder'							:	'./output/bl_outputs/'
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
