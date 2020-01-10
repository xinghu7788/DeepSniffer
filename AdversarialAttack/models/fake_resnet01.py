import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os


downsample_kernel = 1
w_size = 3
channels = [64, 128, 256, 512]
strides = [1, 2, 2, 2]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, if_relu=False):
        global w_size

        self.if_relu = if_relu
        super(BasicBlock, self).__init__()
        if w_size == 3:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if w_size == 3:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #if self.if_relu:                                   #predicted network
            #out = self.relu(out)                           #predicted network

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        global channels, strides

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channels[0], stride=strides[0])
        self.layer2 = self._make_layer(channels[1], stride=strides[1])
        self.layer3 = self._make_layer(channels[2], stride=strides[2])
        #self.layer4 = self._make_layer(channels[3], stride=strides[3], if_relu=True)   # predicted network
        self.layer4 = self._make_layer(channels[3], stride=strides[3], if_relu=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, stride=1, if_relu=False):
        global downsample_kernel

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=downsample_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        layers.append(BasicBlock(self.inplanes, planes, 1, None, if_relu))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def fake_01(pretrained):
    model = ResNet()
    if pretrained != 'null':
        model.load_state_dict(torch.load(pretrained))
    return model


