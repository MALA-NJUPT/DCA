import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import math
import torch.nn.functional as F
import pdb
from torch.distributions.beta import Beta


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


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def Entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def adentropy(f, feat, lamda, coeff=1.0):
    out_t1 = f(feat, reverse=True, coeff=coeff)
    out_t1 = nn.Softmax(dim=1)(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def entropy(f, feat, lamda):
    out_t1 = f(feat)
    softmax_out = nn.Softmax(dim=1)(out_t1)
    loss_ent = lamda * torch.mean(torch.sum(-softmax_out *
                                              (torch.log(softmax_out + 1e-5)), 1))
    return loss_ent

def class_balance(input, lamda):
    msoftmax = input.mean(dim=0)
    loss_div = lamda * torch.sum(msoftmax * (torch.log(msoftmax + 1e-5)))
    return loss_div


def SKL(out1, out2):
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    return (F.kl_div(F.log_softmax(out1, dim=1), out2_t, reduction='none') +
            F.kl_div(F.log_softmax(out2, dim=1), out1_t, reduction='none')) / 2


def mixup_data(images, labels, alpha):
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    lam = Beta(alpha, alpha).sample().to(images.device)
    lam = torch.max(lam, 1 - lam)

    mixed_images = lam.view(-1, 1, 1, 1) * images + (1 - lam).view(-1, 1, 1, 1) * shuffled_images
    mixed_labels = lam * labels + (1 - lam) * shuffled_labels

    return mixed_images, mixed_labels