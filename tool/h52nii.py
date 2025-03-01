import os

import h5py
import numpy as np
import nibabel as nib


def h5_to_nifti(h5_file_path, nifti_file_path):
    # 读取HDF5文件
    with h5py.File(h5_file_path, 'r') as h5_file:
        # 获取'image'和'label'数据集
        if 'image' not in h5_file:
            raise ValueError("HDF5 file should contain a dataset named 'image'")
        # if 'label' not in h5_file:
        #     raise ValueError("HDF5 file should contain a dataset named 'label'")

        image_data = h5_file['image'][:]
        # label_data = h5_file['label'][:]

        # 转换为NIfTI格式
        #
        image_data = np.transpose(image_data, (2, 1, 0))
        # label_data = np.transpose(label_data, (2, 1, 0))

        # 创建NIfTI图像对象
        nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
        # nifti_label_img = nib.Nifti1Image(label_data, affine=np.eye(4))

        # 保存为NIfTI文件
        nib.save(nifti_img, nifti_file_path)
        # nib.save(nifti_label_img, nifti_label_file_path)

path = "E:/WSL4MIS/data/ACDC/ACDC_training_volumes"
output_path='E:/WSL4MIS/code/new_volumes'
h5_list = os.listdir(path)
for i in h5_list:
    h5_file_path = os.path.join(path, i)
    output_name=i.split('.')[0]+'.nii.gz'
    nifti_file_path = os.path.join(output_path, output_name)
    h5_to_nifti(h5_file_path, nifti_file_path)