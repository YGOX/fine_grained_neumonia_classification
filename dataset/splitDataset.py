'''
returns indices for fixed train, valid and test dataset
'''
import h5py
import pandas as pd
import numpy as np
import itertools
import pickle
import os
from configurations.paths import paths, file_names

def getTrainValidTestSplit(label_file, id_file, shuffle, random_seed, train_size=0.8, valid_size=0.1):
    # Read MRI to next label mapping as dataframe
    f = h5py.File(label_file, 'r',  libver='latest')
    pid = pickle.load(open(id_file, 'rb'))
    # Find the set of RID (patients)
    id_keys = list(set(f['id']))
    print(len(id_keys))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(id_keys)

    # Split RID into train:valid:test in 60:20:20 ratio
    train_split_end = int(np.floor(train_size * len(id_keys)))
    print(0, train_split_end)
    valid_split_end = train_split_end + int(np.floor(valid_size * len(id_keys)))
    print(train_split_end, valid_split_end)

    train_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[:train_split_end]]))
    valid_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[train_split_end:valid_split_end]]))
    test_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[valid_split_end:]]))

    print(
    'Total number of patients (train + valid + test) :{}\nPatient count in train set:{}\nPatient count in valid '
    'set:{}\nPatient count in test set:{}\nImages count in Train set:{}\nImages count in Valid set:{},\nImages count in '
    'Test set:{}'.format(
    len(id_keys), len(id_keys[:train_split_end]), len(id_keys[train_split_end:valid_split_end]),
    len(id_keys[valid_split_end:]), len(train_idx), len(valid_idx), len(test_idx)))

    # Save indices of each set in 3dMRItoNextLabel.csv file
    pkl_file = open(os.path.join(paths['data']['datainfo_path'], file_names['data']['Train_data_indices']),
    'wb')
    pickle.dump(train_idx, pkl_file)
    pkl_file.close()

    pkl_file = open(os.path.join(paths['data']['datainfo_path'], file_names['data']['Valid_data_indices']),
    'wb')
    pickle.dump(valid_idx, pkl_file)
    pkl_file.close()

    pkl_file = open(os.path.join(paths['data']['datainfo_path'], file_names['data']['Test_data_indices']),
    'wb')
    pickle.dump(test_idx, pkl_file)
    pkl_file.close()


def getIndicesTrainValidTest(requireslen=False):
    train_indices = pickle.load(open(os.path.join(paths['data']['datainfo_path'],
                                                  file_names['data']['Train_data_indices']), 'rb'))

    valid_indices = pickle.load(open(os.path.join(paths['data']['datainfo_path'],
                                                  file_names['data']['Valid_data_indices']), 'rb'))

    test_indices = pickle.load(open(os.path.join(paths['data']['datainfo_path'],
                                                 file_names['data']['Test_data_indices']), 'rb'))
    if requireslen == True:
        return len(train_indices), len(valid_indices), len(test_indices)
    else:
        return train_indices, valid_indices, test_indices

# Run only when new splits are to be created
getTrainValidTestSplit(os.path.join(paths['data']['datainfo_path'], file_names['data']['data_hdf5_file']),
                       id_file=os.path.join(paths['data']['datainfo_path'], file_names['data']['IDtodict']),
                       shuffle=True,
                       random_seed=200)


def getTrainValidTestSplit1(label_file, id_file, shuffle, random_seed, train_size=0.8, valid_size=0.1):
    # Read MRI to next label mapping as dataframe
    f = h5py.File(label_file, 'r',  libver='latest')
    pid = pickle.load(open(id_file, 'rb'))
    # Find the set of RID (patients)
    id_keys = list(set(f['id']))
    print(len(id_keys))

    if shuffle:
        #np.random.seed(random_seed)
        np.random.shuffle(id_keys)

    # Split RID into train:valid:test in 60:20:20 ratio
    train_split_end = int(np.floor(train_size * len(id_keys)))
    print(0, train_split_end)
    valid_split_end = train_split_end + int(np.floor(valid_size * len(id_keys)))
    print(train_split_end, valid_split_end)

    train_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[:train_split_end]]))
    valid_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[train_split_end:valid_split_end]]))
    test_idx = list(itertools.chain.from_iterable(
    [pid[key] for key in id_keys[valid_split_end:]]))

    print(
    'Total number of patients (train + valid + test) :{}\nPatient count in train set:{}\nPatient count in valid '
    'set:{}\nPatient count in test set:{}\nImages count in Train set:{}\nImages count in Valid set:{},\nImages count in '
    'Test set:{}'.format(
    len(id_keys), len(id_keys[:train_split_end]), len(id_keys[train_split_end:valid_split_end]),
    len(id_keys[valid_split_end:]), len(train_idx), len(valid_idx), len(test_idx)))

    # Save indices of each set in 3dMRItoNextLabel.csv file
    return train_idx, valid_idx, test_idx



