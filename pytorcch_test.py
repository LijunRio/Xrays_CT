import argparse
import os
import sys
import shutil
import time
import random
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

ct_list = os.listdir("D:\\7_9data\\trainning\\ct")
drr_list = list(map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), ct_list))
print(drr_list)
print(ct_list)
ct_list = list(map(lambda x: os.path.join("D:\\7_9data\\trainning\\ct", x), ct_list))
drr_list = list(map(lambda x: os.path.join("D:\\7_9data\\trainning\\seg", x), drr_list))
print(drr_list)
print(ct_list)

img = Image.open('D:\\7_9data\\DRR\\0\\0-1.png').resize((128, 128))
print(np.array(img).shape)
ct_model =sitk.ReadImage("D:\\7_9data\\test.nii", sitk.sitkInt16)
ct_array = sitk.GetArrayFromImage(ct_model)

drr_model = sitk.ReadImage('D:\\7_9data\\DRR\\0\\0-1.png', sitk.sitkInt16)
drr_array = sitk.GetArrayFromImage(drr_model)
print(drr_array.shape)
print(ct_array.shape)
ct_array = torch.FloatTensor(ct_array)
drr_array = torch.FloatTensor(drr_array)
print(drr_array.shape)
print(ct_array.shape)
# seg_array = torch.FloatTensor(seg_array)

#
# transform = transforms.Compose([
#     transforms.ToTensor(),  # convert range [0, 255] to range [0, 1]
# ])
# images = np.zeros((256, 256, 2), dtype=np.uint8)  ### input image size (H, W, C)
# for view_idx in range(2):
#     path = os.path.join('D:\\7_9data\\DRR\\0', '0-{}.png'.format(view_idx))
#     img = Image.open(path).resize((256, 256))
#     images[:, :, view_idx] = np.array(img)
# print(images.shape)
# images = transform(images)
# print(images)

import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
#
# path = "D:\\7_9data\\CT\\0'"
# reader = sitk.ImageSeriesReader()
# dicom_names = reader.GetGDCMSeriesFileNames("D:\\7_9data\\CT\\0'")
# reader.SetFileNames(dicom_names)
# image2 = reader.Execute()
# image_array = sitk.GetArrayFromImage(image2) # z, y, x
# origin = image2.GetOrigin() # x, y, z
# spacing = image2.GetSpacing() # x, y, z
# image3=sitk.GetImageFromArray(image3)##其他三维数据修改原本的数据，
# sitk.WriteImage(image3,'test.nii') #这里可以直接换成image2 这样就保存了原来的数据成了nii格式了。
