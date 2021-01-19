import argparse
import os
import sys
import shutil
import time
import random
import numpy as np

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

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from net import reconnet

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

plt.switch_backend('agg')

# 参数传入器-训练模式
parser = argparse.ArgumentParser(description='PyTorch 3D Reconstruction Training')
parser.add_argument('--exp', type=int, default=1,
                    help='experiments index')
# GPU随机种子-随机参数
parser.add_argument('--seed', type=int, default=1,
                    metavar='N', help='manual seed for GPUs to generate random numbers')
# 半监督学习类别
parser.add_argument('--num-views', type=int, default=1,
                    help='number of classes for semi-supervised training')
# 输入维度
parser.add_argument('--input-size', type=int, default=128,
                    help='dimension of input view size')
# 输出维度
parser.add_argument('--output-size', type=int, default=128,
                    help='dimension of ouput 3D model size')
# 输出结果通道
parser.add_argument('--output-channel', type=int, default=0,
                    help='dimension of ouput 3D model size')
# 模型开始的切片
parser.add_argument('--start-slice', type=int, default=0,
                    help='the idx of start slice in 3D model')
# 测试机的数量
parser.add_argument('--test', type=int, default=1,
                    help='number of total testing samples')
# 三维模型的可视化平面
parser.add_argument('--vis_plane', type=int, default=0,
                    help='visualization plane of 3D images: [0,1,2]')


def main():
    global args
    global exp_path
    args = parser.parse_args()
    # exp_path = './exp{}'.format(args.exp)
    exp_path = './exp'

    # set random seed for GPUs for reproducible
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 它由两个类别
    # 如果是腹部CT就是46深度
    # 如果是肺炎CT就是168深度
    print('Testing Experiement {} ...................'.format(args.exp))
    if args.exp == 1:
        args.output_channel = 46
    elif args.exp == 2:
        args.output_channel = 168
    else:
        assert False, print('Not legal experiment index!')

    # define model
    model = reconnet(in_channels=args.num_views, out_channels=args.output_channel)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function
    criterion = nn.MSELoss(size_average=True, reduce=True).cuda()

    # enable CUDNN benchmark
    cudnn.benchmark = True

    # customized dataset
    # 定制的数据集
    # class(object)类的继承
    class MedReconDataset(Dataset):
        """ 3D Reconstruction Dataset."""

        def __init__(self, csv_file=None, data_dir=None, transform=None):
            self.transform = transform

        def __len__(self):
            return 1

        def __getitem__(self, idx):  # 装载数据，返回[img,label]，idx就是一张一张地读取
            images = np.zeros((args.input_size, args.input_size, args.num_views),
                              dtype=np.uint8)  ### input image size (H, W, C)
            ### load image
            for view_idx in range(args.num_views):
                view_path = os.path.join(exp_path, 'data/2D_projection_{}.jpg'.format(view_idx + 1))
                ### resize 2D images
                img = Image.open(view_path).resize((args.input_size, args.input_size))
                images[:, :, view_idx] = np.array(img)
            if self.transform:
                images = self.transform(images)  # 归一化到 [0.0,1.0]

            ### load target
            model_path = os.path.join(exp_path, 'data/3D_CT.bin')
            model = np.fromfile(model_path, dtype=np.float32)
            model = np.reshape(model, (-1, args.output_size, args.output_size))
            # model = model[ args.start_slice : (args.start_slice + args.output_channel), :, :]

            ### scaling normalize
            model = model - np.min(model)
            model = model / np.max(model)
            labels = torch.from_numpy(model)

            return (images, labels)

    normalize = transforms.Normalize(mean=[0.516], std=[0.264])
    test_dataset = MedReconDataset(
        transform=transforms.Compose([  # 初始化transform模块
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # load model 
    ckpt_file = os.path.join(exp_path, 'model\\model.pth.tar')
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(ckpt_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_file))

    # test evaluation 
    loss, pred_data = test(test_loader, model, criterion, mode='Test')

    # show image
    # pred_data = np.load(os.path.join(exp_path, 'test_results/test_prediction.npz'))
    for idx in range(args.test):
        save_path = os.path.join(exp_path, 'result/sample_{}'.format(idx + 1))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('Evaluate testing sample {} *********************************'.format(idx + 1))
        print("Save prediction to: {}".format(save_path))
        pred = getPred(pred_data, idx)
        groundtruth = getGroundtruth(idx, normalize=True)
        getErrorMetrics(im_pred=pred, im_gt=groundtruth)
        imageSave(pred, groundtruth, args.vis_plane, save_path)

    return


def test(val_loader, model, criterion, mode):
    model.eval()
    losses = AverageMeter()
    pred = np.zeros((args.test, args.output_channel, args.output_size, args.output_size), dtype=np.float32)
    for i, (input, target) in enumerate(val_loader):
        input_var, target_var = Variable(input), Variable(target)
        input_var, target_var = input_var.cuda(), target_var.cuda()

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))
        pred[i, :, :, :] = output.data.float()

        print('{0}: [{1}/{2}]\t'
              'Val Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
            mode, i, len(val_loader), loss=losses))
    print('Average {} Loss: {y:.5f}\t'.format(mode, y=losses.avg))

    save_path = os.path.join(exp_path, 'result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, 'test_prediction.npz')
    print("=> saving test prediction results: '{}'".format(file_name))
    np.savez(file_name, pred=pred)

    return losses.avg, pred


def dataNormalize(data):
    # scaling to [0,1]
    data = data - np.min(data)
    data = data / np.max(data)
    assert ((np.max(data) - 1.0 < 1e-3) and (np.min(data) < 1e-3))
    return data


def getPred(data, idx, test_idx=None):
    pred = data[idx, ...]
    return pred


def getGroundtruth(idx, normalize=True):
    model_path = os.path.join(exp_path, 'data/3D_CT.bin')
    ct3d = np.fromfile(model_path, dtype=np.float32)
    ctslices = np.reshape(ct3d, (-1, args.output_size, args.output_size))
    if normalize:
        ctslices = dataNormalize(ctslices)
    return ctslices


def getErrorMetrics(im_pred, im_gt, mask=None):
    im_pred = np.array(im_pred).astype(np.float)
    im_gt = np.array(im_gt).astype(np.float)
    # sanity check
    assert (im_pred.flatten().shape == im_gt.flatten().shape)
    # RMSE
    rmse_pred = compare_nrmse(im_true=im_gt, im_test=im_pred)
    # PSNR
    psnr_pred = compare_psnr(im_true=im_gt, im_test=im_pred)
    # SSIM
    ssim_pred = compare_ssim(X=im_gt, Y=im_pred)
    # MSE
    mse_pred = mean_squared_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    # MAE
    mae_pred = mean_absolute_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    print("Compare prediction with groundtruth CT:")
    print(
        'mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
            .format(mae_pred=mae_pred, mse_pred=mse_pred, rmse_pred=rmse_pred, psnr_pred=psnr_pred,
                    ssim_pred=ssim_pred))
    return mae_pred, mse_pred, rmse_pred, psnr_pred, ssim_pred


def imageSave(pred, groundtruth, plane, save_path):
    seq = range(pred.shape[plane])
    for slice_idx in seq:
        if plane == 0:
            pd = pred[slice_idx, :, :]
            gt = groundtruth[slice_idx, :, :]
        elif plane == 1:
            pd = pred[:, slice_idx, :]
            gt = groundtruth[:, slice_idx, :]
        elif plane == 2:
            pd = pred[:, :, slice_idx]
            gt = groundtruth[:, :, slice_idx]
        else:
            assert False
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(pd, interpolation='none', cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        f.add_subplot(1, 3, 2)
        plt.imshow(gt, interpolation='none', cmap='gray')
        plt.title('Groundtruth')
        plt.axis('off')
        f.add_subplot(1, 3, 3)
        plt.imshow(gt - pd, interpolation='none', cmap='gray')
        plt.title('Difference image')
        plt.axis('off')
        f.savefig(os.path.join(save_path, 'Plane_{}_ImageSlice_{}.png'.format(plane, slice_idx + 1)))
        plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
