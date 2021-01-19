import torch.nn as nn
import torch
import math
from torch.nn import init


# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv2x2(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=2, stride=stride, padding=0,
                     bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=1, stride=stride, padding=0,
                     bias=False)


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv4x4x4(in_planes, out_planes, stride=2):
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=4, stride=stride, padding=1,
                     bias=False)


# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv2x2(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=2, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=1, stride=stride, padding=0, output_padding=0,
                              bias=False)


def deconv3x3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=0,
                              bias=False)


def deconv4x4x4(in_planes, out_planes, stride):
    return nn.ConvTranspose3d(in_planes, out_planes,
                              kernel_size=4, stride=stride, padding=1, output_padding=0,
                              bias=False)


def _make_layers(in_channels, output_channels, type, batch_norm=False, activation=None):
    layers = []

    if type == 'conv1_s1':
        layers.append(conv1x1(in_channels, output_channels, stride=1))
    elif type == 'conv2_s2':
        layers.append(conv2x2(in_channels, output_channels, stride=2))
    elif type == 'conv3_s1':
        layers.append(conv3x3(in_channels, output_channels, stride=1))
    elif type == 'conv4_s2':
        # def conv4x4(in_planes, out_planes, stride=2):
        # 	return nn.Conv2d(in_planes, out_planes,
        # 					 kernel_size=4, stride=stride, padding=1,
        # 					 bias=False)
        layers.append(conv4x4(in_channels, output_channels, stride=2))
    elif type == 'deconv1_s1':
        layers.append(deconv1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv2_s2':
        layers.append(deconv2x2(in_channels, output_channels, stride=2))
    elif type == 'deconv3_s1':
        layers.append(deconv3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4_s2':
        layers.append(deconv4x4(in_channels, output_channels, stride=2))
    elif type == 'conv1x1_s1':
        layers.append(conv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv1x1_s1':
        layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
    elif type == 'deconv3x3_s1':
        layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
    elif type == 'deconv4x4_s2':
        layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
    else:
        raise NotImplementedError('layer type [{}] is not implemented'.format(type))

    if batch_norm == '2d':
        layers.append(nn.BatchNorm2d(output_channels))
    elif batch_norm == '3d':
        layers.append(nn.BatchNorm3d(output_channels))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'sigm':
        layers.append(nn.Sigmoid())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU(0.2, True))
    else:
        if activation is not None:
            raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

    return nn.Sequential(*layers)


def _init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                print('Initializing Weights: {}...'.format(classname))
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Sequential') == -1 and classname.find('Conv5_Deconv5_Local') == -1:
            raise NotImplementedError('initialization of [{}] is not implemented'.format(classname))

    print('initialize network with {}'.format(init_type))
    net.apply(init_func)


def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class ReconNet(nn.Module):

    def __init__(self, in_planes, out_planes, gain=0.02, init_type='standard'):
        super(ReconNet, self).__init__()

        # 表征层
        ######### representation network - convolution layers
        # 理论上来说input-channel为1 ，那么就是1->256  kernel为
        # type == 'conv4_s2': layers.append(conv4x4(in_channels, output_channels, stride=2))
        self.conv_layer1 = _make_layers(in_planes, 256, 'conv4_s2', False)
        self.conv_layer2 = _make_layers(256, 256, 'conv3_s1', '2d')
        # 参数： inplace-选择是否进行覆盖运算
        # 的意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
        self.relu2 = nn.ReLU(inplace=True)

        self.conv_layer3 = _make_layers(256, 512, 'conv4_s2', '2d', 'relu')
        self.conv_layer4 = _make_layers(512, 512, 'conv3_s1', '2d')
        self.relu4 = nn.ReLU(inplace=True)

        self.conv_layer5 = _make_layers(512, 1024, 'conv4_s2', '2d', 'relu')
        self.conv_layer6 = _make_layers(1024, 1024, 'conv3_s1', '2d')
        self.relu6 = nn.ReLU(inplace=True)

        self.conv_layer7 = _make_layers(1024, 2048, 'conv4_s2', '2d', 'relu')
        self.conv_layer8 = _make_layers(2048, 2048, 'conv3_s1', '2d')
        self.relu8 = nn.ReLU(inplace=True)

        self.conv_layer9 = _make_layers(2048, 4096, 'conv4_s2', '2d', 'relu')
        self.conv_layer10 = _make_layers(4096, 4096, 'conv3_s1', '2d')
        self.relu10 = nn.ReLU(inplace=True)

        ######### transform module
        self.trans_layer1 = _make_layers(4096, 4096, 'conv1_s1', False, 'relu')
        self.trans_layer2 = _make_layers(2048, 2048, 'deconv1x1_s1', False, 'relu')

        ######### generation network - deconvolution layers
        self.deconv_layer10 = _make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer8 = _make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer7 = _make_layers(512, 512, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer6 = _make_layers(512, 256, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer5 = _make_layers(256, 256, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer4 = _make_layers(256, 128, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer3 = _make_layers(128, 128, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer2 = _make_layers(128, 64, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer1 = _make_layers(64, 64, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer0 = _make_layers(64, 1, 'conv1x1_s1', False, 'relu')
        self.output_layer = _make_layers(64, out_planes, 'conv1_s1', False)

        if init_type == 'standard':
            _initialize_weights(self)
        else:
            _init_weights(self, gain=gain, init_type=init_type)

    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        ### representation network
        conv1 = self.conv_layer1(x)
        conv2 = self.conv_layer2(conv1)
        relu2 = self.relu2(conv1 + conv2)
        conv3 = self.conv_layer3(relu2)
        conv4 = self.conv_layer4(conv3)
        relu4 = self.relu4(conv3 + conv4)
        conv5 = self.conv_layer5(relu4)
        conv6 = self.conv_layer6(conv5)
        relu6 = self.relu6(conv5 + conv6)
        conv7 = self.conv_layer7(relu6)
        conv8 = self.conv_layer8(conv7)
        relu8 = self.relu8(conv7 + conv8)
        conv9 = self.conv_layer9(relu8)
        conv10 = self.conv_layer10(conv9)
        relu10 = self.relu10(conv9 + conv10)

        ### transform module
        features = self.trans_layer1(relu10)
        # 相当于numpy中的resize
        # -1表示我们不想自己计算，电脑帮我们计算出对应的数字
        trans_features = features.view(-1, 2048, 2, 4, 4)
        trans_features = self.trans_layer2(trans_features)

        ### generation network
        deconv10 = self.deconv_layer10(trans_features)
        deconv8 = self.deconv_layer8(deconv10)
        deconv7 = self.deconv_layer7(deconv8)
        deconv6 = self.deconv_layer6(deconv7)
        deconv5 = self.deconv_layer5(deconv6)
        deconv4 = self.deconv_layer4(deconv5)
        deconv3 = self.deconv_layer3(deconv4)
        deconv2 = self.deconv_layer2(deconv3)
        deconv1 = self.deconv_layer1(deconv2)

        ### output
        out = self.deconv_layer0(deconv1)
        # torch.squeeze(input, dim=None, out=None) → Tensor的用法主要就是对数据的维度进行压缩或者解压
        # input (Tensor) – the input tensor.
        # dim (int, optional) – if given, the input will be squeezed only in this dimension
        # out (Tensor, optional) – the output tensor.
        out = torch.squeeze(out, 1)
        out = self.output_layer(out)

        return out


def reconnet(in_channels, out_channels, **kwargs):
    model = ReconNet(in_channels, out_channels, **kwargs)
    return model

