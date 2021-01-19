import torch
import SimpleITK as sitk
from dataset import MedReconDataset
import os
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


def singal_hist_img_show(data_hu):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(data_hu.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.imshow(data_hu)
    plt.show()


result_ct = "D:\\Rio\\test_nature\\test3.nii"
out = "D:\\Rio\\test_nature\\mask.nii"
ct_img = sitk.ReadImage(result_ct)
ct_array = sitk.GetArrayFromImage(ct_img)
ct_array.astype(np.int16)

ct_array[ct_array > 100] = 255
ct_array[ct_array <= 100] = 0

ct_brinary = sitk.GetImageFromArray(ct_array)

ct_brinary.SetSpacing([1, 1, 1.25])
sitk.WriteImage(ct_brinary, out)
#
# singal_hist_img_show(ct_array[10])
# plt.show()
