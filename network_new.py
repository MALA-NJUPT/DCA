import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable, Function
from utils import *
import math, pdb
import torch.nn.utils.weight_norm as weightNorm


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.coeff
        return output, None


def grad_reverse(x, coeff=1.0):
    return GradientReverseFunction.apply(x, coeff)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d, }


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class classifier_C(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="dca"):
        super(classifier_C, self).__init__()
        self.type = type
        if type == "dca":
            self.fc1 = weightNorm(nn.Linear(bottleneck_dim, bottleneck_dim), name="weight")
            self.fc1.apply(init_weights)
            self.fc2 = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc2.apply(init_weights)


    def forward(self, x, reverse=False, coeff=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, coeff)
        x = self.fc2(x)
        return x


class classifier_D(nn.Module):
    def __init__(self, feature_dim, class_num, type="dca"):
        super(classifier_D, self).__init__()
        self.type = type
        if type == "dca":
            # self.bn = nn.BatchNorm1d(feature_dim, affine=True)
            # self.relu = nn.ReLU(inplace=True)
            # self.dropout = nn.Dropout(p=0.5)
            self.fc1 = weightNorm(nn.Linear(feature_dim, feature_dim), name="weight")
            self.fc1.apply(init_weights)
            self.fc2 = weightNorm(nn.Linear(feature_dim, class_num), name="weight")
            self.fc2.apply(init_weights)


    def forward(self, x, reverse=False, coeff=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, coeff)
        x = self.fc2(x)
        return x


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    def __init__(self, input_dim=100, input_size=224, class_num=10, batch_size=64):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.class_num = class_num
        self.batch_size = batch_size

        # label embedding
        self.label_emb = nn.Embedding(self.class_num, self.input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            # nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.Linear(1024, 128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Tanh(),
        )
        init_weights(self)

    def forward(self, input, label):
        x = torch.mul(self.label_emb(label), input)
        # x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 512, (self.input_size // 32), (self.input_size // 32))
        x = self.deconv(x)
        x = x.view(x.size(0), -1)

        return x