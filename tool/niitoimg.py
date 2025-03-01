import os
import nibabel as nib
import numpy as np
from PIL import Image

def convert_nii_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # 构建完整的文件路径
            file_path = os.path.join(input_folder, filename)
            # 读取 NIfTI 文件
            nii_image = nib.load(file_path)
            image_data = nii_image.get_fdata()

            # 选择中间的切片
            slice_idx = image_data.shape[2] // 2
            slice_data = image_data[:, :, slice_idx]

            # 转换为8位图像格式
            slice_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
            image_8bit = (slice_normalized * 255).astype(np.uint8)
            image = Image.fromarray(image_8bit)

            # 保存图像
            output_filename = filename.replace('.nii.gz', '.png')
            image.save(os.path.join(output_folder, output_filename))
            print(f"Converted {filename} to {output_filename}")


if __name__=='__main__':

    input_folder = './ce6190 as2/06'  # nii.gz文件路径
    output_folder = './06' # png文件路径
    if os.path.exists(output_folder)==False:
        os.makedirs(output_folder)

    convert_nii_to_png(input_folder, output_folder)