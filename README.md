# fine_grained_neumonia_classification

weakly supervised detector for recognition of multitypes of neumonia

Support Multi-GPU training 

Download Data from:
链接:https://pan.baidu.com/s/178v4J5z4NIT_kxfz4-8yIg  密码:pjho
Uzip the stage1.zip you will get patient folders including their DICOMs

Data Proprecessing:
run python prep.py-> resulting a folder 'prep_result1' including patient CT array:*****_clean.npy, Patient specific Lung Segmentation Mask: *****_label.npy.

Sampling Axial slices and put patient ID, disease label into a h5py file (resulting a h5py file named data.hdf5) :
run python dataset/createh5pyData.py 

Generate Patient Specific Index (resulting a pickle file named IDtodict.pkl): 
run python dataset/patientmapping.py 

Splite data into train, validation and test (resulting train_data_indices.pkl (train images indices) valid_data_indices.pkl (validation images indices) test_data_indices.pkl (test images indices): 
run python dataset/splitDataset.py 

For training and validating 
run -> run.sh 

