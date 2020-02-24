#preprocessing and save data in .npy

from preprocessing.full_prep import full_prep
from configurations.paths import paths, file_names

datapath = paths['data']['datapath']
prep_result_path =paths['data']['preprocess_result_path']

testsplit= full_prep(datapath,prep_result_path)


