import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
# save images in slice level
import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

# saving images in slice level


slice_num = 0
mask_path = sorted(glob.glob("/home1/luyi/WSL4MIS/data/MSCMR/*_manual.nii.gz"))

for case in mask_path:
    print("case:", case)

    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)

    image_path = case.replace("_manual", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)

    scribble_path = case.replace("_manual", "_scribble")
    scribble_itk = label_itk
    scribble = sitk.GetArrayFromImage(scribble_itk)

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = case.split("/")[-1].split(".")[0].replace("_manual", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    for slice_ind in range(image.shape[0]):
        f = h5py.File('/home1/luyi/WSL4MIS/data/MSCMR/{}_slice_{}.h5'.format(item, slice_ind), 'w')
        f.create_dataset('image', data=image[slice_ind], compression="gzip")
        f.create_dataset('scribble', data=scribble[slice_ind], compression="gzip")
        # f.create_dataset(
        #     'scribble', data=scribble[slice_ind], compression="gzip")
        f.close()
        slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))

# saving images in volume level


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())

