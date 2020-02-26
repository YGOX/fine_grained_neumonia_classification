'''
returns dictionary of patients mapping
'''

import h5py
import numpy as np
import pickle
import os
from configurations.paths import paths, file_names

fname = os.path.join(paths['data']['datainfo_path'], file_names['data']['data_hdf5_file'])
pkl_file = open(os.path.join(paths['data']['datainfo_path'], file_names['data']['IDtodict']), 'wb')

f = h5py.File(fname, 'r')
print (f.keys())
IDtodict = dict()
id_set = set(f['id'])

for index in id_set:
    indx = np.where(f['id'][:] == index)[0]
    IDtodict[index] = indx

pickle.dump(IDtodict, pkl_file)
pkl_file.close()

f.close()