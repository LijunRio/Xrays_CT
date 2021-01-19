import os
from time import time

from torchsummary import summary
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import MedReconDataset
from net import reconnet
import torchvision.transforms as transforms

# 设置显卡相关
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cudnn.benchmark = True
device_ids = [1]
device = torch.device('cuda:1')

# 加载数据
train_set_path = "D:\\7_15data\\binary_train"
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

tran_dataset = MedReconDataset(os.path.join(train_set_path, 'drr'), os.path.join(train_set_path, 'ct'))
train_dl = DataLoader(tran_dataset, 1, True, num_workers=0, pin_memory=True)

# 定义网络
model = reconnet(in_channels=1, out_channels=128)
model.cuda()
# criterion = torch.nn.MSELoss(size_average=True, reduce=True).cuda()
criterion = torch.nn.MSELoss(reduction='mean').cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)

print('---------- Networks initialized -------------')
summary(model, (1, 128, 128))
print('-----------------------------------------------')

start = time()
# epoch = 10000
# freeze_support()
for epoch in tqdm(range(100)):
    running_loss = 0.0
    for step, sample in enumerate(train_dl):
        drr = sample['drr']
        ct = sample['ct']
        drr = drr.cuda()
        ct = ct.cuda()

        optimizer.zero_grad()
        # outputs = model(drr)

        try:
            outputs = model(drr)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        loss = criterion(outputs, ct)
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()
        if step % 10 == 9:  # 每十次计算一次loss
            print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 10))
            running_loss = 0.0

    print('Saving epoch %d model ...' % (epoch + 1))
    # 参数保存###########
    state = {
        'net': model.state_dict(),
        'epoch': epoch + 1,
    }  # 1 、 先建立一个字典
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')  # 2 、 建立一个保存参数的文件夹
    # torch.save(state, './checkpoint/reconnect_epoch_%d.ckpt' % (epoch + 1))  # 3 、保存操作
    # 因为在for epoch in range（num_epoch）这个循环中，所以可以 保存每一个epoch的参数，如果不在这个循环中，
    # 而是循环完成在保存，则保存的是最后一个epoch的参数
torch.save(model, 'binary_train.pth') # save net model and parameters
print('Finished Training')
