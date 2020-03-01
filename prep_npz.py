from preprocessing.full_prep_npz import full_prep
from configurations.paths import paths, file_names
import os
datapath = os.path.join(paths['data']['raw_data'], '*.npz')
#datapath = paths['data']['datapath']
prep_result_path =paths['data']['preprocess_result_path']

testsplit= full_prep(datapath, prep_result_path)

