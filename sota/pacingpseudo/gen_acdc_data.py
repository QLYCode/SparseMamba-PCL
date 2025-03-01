
import cv2
import numpy as np


x = np.load("/home/linux/Desktop/WSL4MIS/seg00/pacingpseudo/data/chaos/train_test_split/data_2d/t1/1/patient1_t1_inphase_3.npz")
print(x.files)
print(x["uid"])
print(x["img"].shape, x["img"].dtype)
print(x["lab"].shape, x["lab"].dtype)
print(x["scb"].shape, x["scb"].dtype)

print(np.unique(x["img"]))
print(np.unique(x["lab"]))
print(np.unique(x["scb"]))


print("==========================")

import nibabel as nib
import os
import cv2
import random
from glob import glob

imagefs = sorted(glob("/home/linux/Desktop/WSL4MIS/data/ACDC/raw/images/*.nii.gz"))

save_root = "/home/linux/Desktop/WSL4MIS/seg00/pacingpseudo/data/acdc/train_test_split/data_2d/Cine"

# 10% 作为测试集
# 不做交叉验证划分数据集，但仍然保存原来的形式 记录在txt文件中
train_f = open("/home/linux/Desktop/WSL4MIS/seg00/pacingpseudo/data/acdc/train_test_split/five_fold_split/train_fold0.txt", 'w', encoding="utf-8")
test_f = open("/home/linux/Desktop/WSL4MIS/seg00/pacingpseudo/data/acdc/train_test_split/five_fold_split/test_fold0.txt", 'w', encoding="utf-8")

for imgf in imagefs:
    bname = os.path.basename(imgf)
    if bname == "patient001_frame01_9anno.nii.gz":
        continue

    # 获取uid
    uid = str(bname).replace(".nii.gz", "")
    isTrain = False if random.random() < 0.2 else True
    print(bname, uid, f"is train:{isTrain}")

    # 读取图像
    img = nib.load(imgf)
    img_data = img.get_fdata()

    # 读取label
    labelf = os.path.join("/home/linux/Desktop/WSL4MIS/data/ACDC/raw/full_annos", bname)
    assert os.path.exists(labelf)
    label = nib.load(labelf)
    label_data = label.get_fdata()
    assert img_data.shape == label_data.shape

    # 读取 scribbles
    scribf = os.path.join("/home/linux/Desktop/WSL4MIS/data/ACDC/raw/scribbles", bname)
    assert os.path.exists(scribf)
    scrib = nib.load(scribf)
    scrib_data = scrib.get_fdata()
    assert img_data.shape == scrib_data.shape

    nums = img_data.shape[-1]
    for i in range(nums):
        timg = cv2.resize(img_data[:, :, i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        tlab = cv2.resize(label_data[:, :, i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        tscb = cv2.resize(scrib_data[:, :, i], dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        tlab = np.asarray(tlab, dtype=np.int64)
        tscb = np.asarray(tscb, dtype=np.int64)

        tscb[np.where(tscb == 4)] = 0
        # print(timg.dtype)
        # print(np.unique(tlab), tlab.dtype)
        # print(np.unique(tscb), tscb.dtype)
        # quit()

        # print(np.unique(tlab))
        # print(np.unique(tscb))
        # print('---------------')


        save_name = uid + f"_{i}.npz"
        save_path = os.path.join(save_root, save_name)
        np.savez(save_path, uid=uid, img=timg, lab=tlab, scb=tscb)
        print("saved", save_path)

        file = train_f if isTrain else test_f
        content = f"train_test_split/data_2d/Cine/{save_name}\n"
        file.writelines(content)

    # quit()

train_f.close()
test_f.close()

# print(x["uid"])
# print(x["img"].shape)
# print(x["lab"].shape)
# print(x["scb"].shape)
