import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network_new
from network_new import *
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from loss import *
import torch.nn.functional as F
from utils import *


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 5e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def data_load(args):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(args.split * dsize)
        test_size = dsize - tr_size
        print(dsize, tr_size, test_size)
        tr_txt, te_txt = torch.utils.data.random_split(
            txt_src, [tr_size, test_size]
        )
    else:
        tr_txt = txt_src
        te_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=train_transform)
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=test_transform)
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList(txt_test, transform=test_transform)
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network

    netG = network_new.ResBase(res_name=args.net).cuda()
    netF = network_new.bottleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network_new.classifier_C(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netD = network_new.classifier_D(type=args.layer, feature_dim=netG.in_features, class_num=args.class_num).cuda()


    param_group_g = []
    param_group_c = []
    param_group_d = []
    learning_rate = args.lr
    for k, v in netG.named_parameters():  # k是网络层的名字  v是其中的参数
        param_group_g += [{"params": v, "lr": learning_rate * 0.1}]  # 1
    for k, v in netF.named_parameters():  # k是网络层的名字  v是其中的参数
        param_group_g += [{"params": v, "lr": learning_rate * 1.0}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": learning_rate * 1.0}]# 10
    for k, v in netD.named_parameters():
        param_group_d += [{"params": v, "lr": learning_rate * 1.0}]

    # optimizer_g = optim.SGD(param_group_g, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer_c = optim.SGD(param_group_c, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer_d = optim.SGD(param_group_d, momentum=0.9, weight_decay=5e-4, nesterov=True)

    optimizer_g = optim.SGD(param_group_g)
    optimizer_c = optim.SGD(param_group_c)
    optimizer_d = optim.SGD(param_group_d)

    optimizer_g = op_copy(optimizer_g)
    optimizer_c = op_copy(optimizer_c)
    optimizer_d = op_copy(optimizer_d)

    acc_init = 0
    netG.train()
    netF.train()
    netC.train()
    netD.train()
    iter_num = 0

    iter_source = iter(dset_loaders["source_tr"])
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // args.interval

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()
        if inputs_source.size(0) == 1:
            continue
        iter_num += 1

        lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_d, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        features_d = netG(inputs_source)
        features = netF(features_d)


        outputs_source1 = netC(features)
        outputs_source2 = netD(features_d)

        classifier_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source1, labels_source)
        classifier_loss2 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source2, labels_source)

        classifier_loss = 1.0 * classifier_loss1 + 1.0 * classifier_loss2
        all_loss = classifier_loss

        optimizer_g.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        all_loss.backward()
        optimizer_g.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netG.eval()
            netF.eval()
            netC.eval()
            netD.eval()
            acc1, acc_list1, accuracy1 = cal_acc(dset_loaders["source_te"], netG, netF, netC)
            acc2, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["source_te"], netG, netD)
            acc_best = acc1
            log_str = (
                    "Task: {}, Iter:{}; Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}%".format(
                        args.name_src, iter_num, acc1, acc2
                    )
                    + "\n"
                    + str(acc_list1)
            )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            if acc_best >= acc_init:
                acc_init = acc_best
                best_netG = netG.state_dict()
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()
                best_netD = netD.state_dict()

                torch.save(best_netG, osp.join(args.output_dir_src, "source_G.pt"))
                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
                torch.save(best_netD, osp.join(args.output_dir_src, "source_D.pt"))

            netG.train()
            netF.train()
            netC.train()
            netD.train()
    print('Best Model Saved!!')

    return netG, netF, netC, netD


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network

    netG = network_new.ResBase(res_name=args.net).cuda()
    netF = network_new.bottleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network_new.classifier_C(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netD = network_new.classifier_D(type=args.layer, feature_dim=netG.in_features, class_num=args.class_num).cuda()


    args.modelpath = args.output_dir_src + "/source_G" + ".pt"
    netG.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_F" + ".pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_C" + ".pt"
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + "/source_D" + ".pt"
    netD.load_state_dict(torch.load(args.modelpath))

    netG.eval()
    netF.eval()
    netC.eval()
    netD.eval()


    acc1, acc_list1, _ = cal_acc(dset_loaders["test"], netG, netF, netC)
    acc2, acc_list2, _ = cal_acc_easy(dset_loaders["test"], netG, netD)
    log_str = (
        "\nDateset: {}, Task: {}, Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}%".format(args.dset, args.name, acc1, acc2)
        + "\n"
        + str(acc_list1)
    )
    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    print(log_str + "\n")


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCA on visda")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size") # 128
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda", choices=["office", "officehome", "visda"])
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet101", help="resnet50, resnet101")
    parser.add_argument("--seed", type=int, default=2042, help="random seed")
    parser.add_argument("--max_epoch", type=int, default=10, help="max iterations")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="dca", choices=["linear", "wn", "dca"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="double")
    parser.add_argument("--da", type=str, default="SFDA")
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    parser.add_argument("--split", type=float, default=0.9, help="split parameter")

    args = parser.parse_args()

    if args.dset == "office":
        names = ["amazon", "dslr", "webcam"]
        args.class_num = 31
    elif args.dset == "officehome":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    elif args.dset == 'visda':
        names = ['train', 'validation']
        args.class_num = 12


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = "./data/"
    if args.dset == "office":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_31.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_31.txt"
    elif args.dset == "officehome":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_65.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_65.txt"
    elif args.dset == "visda":
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_12.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_12.txt"
    args.test_dset_path = args.t_dset_path

    current_folder = "./ckps/"
    args.output_dir_src = osp.join(
        current_folder, args.da, args.output, args.dset, names[args.s][0].upper()
    )  # ./ckps/uda/bait/office/A
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system("mkdir -p " + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.output_dir = osp.join(
        current_folder,
        args.da,
        args.output,
        args.dset,
        names[args.s][0].upper() + names[args.t][0].upper(),
    )
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir_src, "log.txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    train_source(args)
    test_target(args)
