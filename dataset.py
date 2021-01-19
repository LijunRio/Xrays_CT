import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
from torch.utils.data import DataLoader


class MedReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""

    def __init__(self, drr_dir, ct_dir, transform=None):
        # 获取训练集文件夹下的所有文件
        self.drr_list = os.listdir(drr_dir)
        self.ct_list = os.listdir(ct_dir)
        # 获取每个训练文件的完整路径
        self.drr_list = list(map(lambda x: os.path.join(drr_dir, x), self.drr_list))
        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        # 图像变换
        self.transform = transform

    def __len__(self):
        return len(self.ct_list)

    # 装载数据，返回[img,label]，idx就是一张一张地读取
    def __getitem__(self, idx):
        drr_path = self.drr_list[idx]
        ct_path = self.ct_list[idx]

        drr_img = sitk.ReadImage(drr_path, sitk.sitkInt16)
        ct_model = sitk.ReadImage(ct_path, sitk.sitkInt16)

        drr_array = sitk.GetArrayFromImage(drr_img)
        ct_array = sitk.GetArrayFromImage(ct_model)
        ct_array = ct_array.astype(np.float32)
        # ct_array = ct_array / 200

        # 针对二值图
        ct_array = ct_array * 255

        # 处理完毕，将array转换为tensor
        # images = torch.FloatTensor(drr_array)
        images = torch.FloatTensor(drr_array).unsqueeze(0)
        ct_labels = torch.FloatTensor(ct_array)
        sample = {'drr': images, 'ct': ct_labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


# # 这是一个测试函数,也即我的代码写好后,如果直接python运行当前py文件,就会执行以下代码的内容,以检测我上面的代码是否有问题,这其实就是方便我们调试,而不是每次都去run整个网络再看哪里报错
if __name__ == '__main__':
    train_set_path = "D:\\7_15data\\training"
    tran_dataset = MedReconDataset(os.path.join(train_set_path, 'drr'), os.path.join(train_set_path, 'ct'))
    train_dl = DataLoader(tran_dataset, 1, True, num_workers=0, pin_memory=True)
    for i, sample in enumerate(train_dl):
       print(i, sample['drr'].size(), sample['ct'].size())

# for i in range(100):
#     if i % 10 == 9:
#         print(i)
