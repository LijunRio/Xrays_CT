import torch
import SimpleITK as sitk
from dataset import MedReconDataset
import os
from torch.utils.data import DataLoader

device = torch.device('cuda')
model_path = 'D:/Rio/test_nature/model/binary_train.pth'
drr_path = 'D:/7_15data/training/drr/8.png'
output_path = 'D:/7_15data/new/8_mask.nii'

drr_img = sitk.ReadImage(drr_path, sitk.sitkInt16)
drr_array = sitk.GetArrayFromImage(drr_img)
images = torch.FloatTensor(drr_array).unsqueeze(0)
images = images.unsqueeze(0)
print(images.shape)
#
# train_set_path = "D:\\7_15data\\training"
# tran_dataset = MedReconDataset(os.path.join(train_set_path, 'drr'), os.path.join(train_set_path, 'ct'))
# train_dl = DataLoader(tran_dataset, 1, True, num_workers=0, pin_memory=True)


model = torch.load(model_path)
model.to(device)
with torch.no_grad():
    drr_input = images.to(device)
    out = model(drr_input)
    out_cpu = out.cpu()
    out_array = out_cpu.numpy()
    print(out_array.shape)
    out_seques = out_array.squeeze(0)
    print(out_seques.shape)

    # 二值化
    out_seques[out_seques > 100] = 255
    out_seques[out_seques <= 100] = 0
    # 转为图片
    out_ct = sitk.GetImageFromArray(out_seques)




    
    sitk.WriteImage(out_ct, output_path)



