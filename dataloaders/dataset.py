import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pydicom
import nibabel as nib

from random import sample


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.all_volumes = [i for i in self.all_volumes if '.h5' in i]
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        elif fold == "MAAGfold":
            training_set = ["patient{:0>3}".format(i) for i in
                            [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                             71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in
                            [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                             71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51,
                             40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                             23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = sample(["patient{:0>3}".format(i) for i in
                                   [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                    71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3,
                                    8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                                    23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in
                              [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# class MSCMRDataSets(Dataset):
#     def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.sup_type = sup_type
#         self.transform = transform
#         train_ids, test_ids = self._get_fold_ids(fold)

#         if self.split == 'train':
#             self.all_slices = os.listdir(self._base_dir + "/MSCMR_training_slices")
#             self.sample_list = []
#             for ids in train_ids:
#                 new_data_list = list(filter(lambda x: re.match(
#                     '{}.*'.format(ids), x) != None, self.all_slices))
#                 self.sample_list.extend(new_data_list)

#         elif self.split == 'val':
#             self.all_volumes = os.listdir(self._base_dir + "/MSCMR_training_volumes")
#             self.sample_list = []
#             for ids in test_ids:
#                 new_data_list = list(filter(lambda x: re.match(
#                     '{}.*'.format(ids), x) != None, self.all_volumes))
#                 self.sample_list.extend(new_data_list)

#         # if num is not None and self.split == "train":
#         #     self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))

#     def _get_fold_ids(self, fold):
#         training_set = ["subject{:0>2}".format(i) for i in
#                         [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
#         # validation_set = ["subject{:0>2}".format(i) for i in
#         #                   [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7,
#         #                    9]]
#         validation_set = ["subject{:0>2}".format(i) for i in [1, 29, 36, 41, 8]]
#         return [training_set, validation_set]

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train":
#             h5f = h5py.File(self._base_dir +
#                             "/MSCMR_training_slices/{}".format(case), 'r')
#         else:
#             h5f = h5py.File(self._base_dir +
#                             "/MSCMR_training_volumes/{}".format(case), 'r')
#         image = h5f['image'][:]
#         label = h5f['scribble'][:]
#         sample = {'image': image, 'label': label}

#         if self.split == "train":
#             image = h5f['image'][:]
#             if self.sup_type == "random_walker":
#                 label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
#             else:
#                 label = h5f[self.sup_type][:]
#             sample = {'image': image, 'label': label}
#             sample = self.transform(sample)


#         else:
#             image = h5f['image'][:]
#             # label = h5f[self.sup_type][:]
#             label = h5f['scribble'][:]
#             image=np.float64(image)
#             label=np.float64(label)
#             sample = {'image': image, 'label': label}

#         sample["idx"] = idx

#         return sample


class ChaosDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.data_root = data_root
        self.splits = split
        self.transform = transform
        self.train_ids = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34]
        self.test_ids = [36, 37, 38, 39]

        self.ids = self.train_ids if split == 'train' else self.test_ids
        self.samples = []
        for data_id in self.ids:
            sample_path = os.path.join(data_root, str(data_id), 'T1DUAL/DICOM_anon/InPhase')

            scribble_dir = os.path.join(data_root, str(data_id), 'T1DUAL/Ground_scribble')

            input_files = os.listdir(sample_path)
            label_files = os.listdir(scribble_dir)
            input_files.sort()
            label_files.sort()
            assert len(input_files) == len(label_files), '输入和标签数量不一致'
            for i, input_file in enumerate(input_files):
                input_path = os.path.join(sample_path, input_file)
                label_path = os.path.join(scribble_dir, label_files[i])
                self.samples.append((input_path, label_path))
        print('数据集大小:', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, label_path = self.samples[idx]
        # 输入文件是dicom格式
        input_img = pydicom.dcmread(input_path).pixel_array
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
        label_img = np.where(label_img == 0, 4, label_img)
        label_img = np.where(label_img == 63, 0, label_img)
        label_img = np.where(label_img == 126, 1, label_img)
        label_img = np.where(label_img == 189, 2, label_img)
        label_img = np.where(label_img == 252, 3, label_img)

        sample = {'image': input_img, 'label': label_img}
        if self.transform:
            sample = self.transform(sample)

        sample['idx'] = idx
        return sample


class MSCMRDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.data_root = data_root
        self.splits = split
        self.transform = transform
        if split == 'train':
            self.splits_dir = os.path.join(data_root, 'train')
        else:
            self.splits_dir = os.path.join(data_root, 'TestSet')

        self.samples = []
        images_list = os.listdir(os.path.join(self.splits_dir, 'images'))
        labels_list = os.listdir(os.path.join(self.splits_dir, 'labels'))
        images_list.sort()
        labels_list.sort()
        assert len(images_list) == len(labels_list), '输入和标签数量不一致'
        for i, image_file in enumerate(images_list):
            image_path = os.path.join(self.splits_dir, 'images', images_list[i])
            label_path = os.path.join(self.splits_dir, 'labels', labels_list[i])
            img = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            for j in range(img.shape[2]):
                self.samples.append({'image': img[:, :, j], 'label': label[:, :, j]})

        print('数据集大小:', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_img = self.samples[idx]['image']
        label_img = self.samples[idx]['label']

        # #经过scribble后像素值不是0-4，将label的像素值[63,126,189,252,0]分别转换为[0,1,2,3,4]
        # label_img = np.where(label_img == 0, 4, label_img)
        # label_img = np.where(label_img == 63, 0, label_img)
        # label_img = np.where(label_img == 126, 1, label_img)
        # label_img = np.where(label_img == 189, 2, label_img)
        # label_img = np.where(label_img == 252, 3, label_img)

        sample = {'image': input_img, 'label': label_img}
        if self.transform:
            sample = self.transform(sample)

        sample['idx'] = idx
        return sample