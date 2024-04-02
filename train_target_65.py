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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=train_transform)
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    num_examp = len(dsets["target"])
    dsets["test"] = ImageList(txt_test, transform=test_transform)
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders, num_examp


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target_mme(args):
    dset_loaders, num_examp = data_load(args)
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

    netC.eval()
    netD.train()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group_g = []
    param_group_d = []

    learning_rate = args.lr
    for k, v in netG.named_parameters():
        param_group_g += [{"params": v, "lr": learning_rate * 0.1}]
    for k, v in netF.named_parameters():
        param_group_g += [{"params": v, "lr": learning_rate * 1.0}]
    for k, v in netD.named_parameters():
        param_group_d += [{"params": v, "lr": learning_rate * 2.0}]

    # optimizer_g = optim.SGD(param_group_g, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer_d = optim.SGD(param_group_d, momentum=0.9, weight_decay=5e-4, nesterov=True)

    optimizer_g = optim.SGD(param_group_g)
    optimizer_d = optim.SGD(param_group_d)

    optimizer_g = op_copy(optimizer_g)
    optimizer_d = op_copy(optimizer_d)

    iter_num = 0
    iter_target = iter(dset_loaders["target"])
    max_iter = (args.max_epoch) * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    # len(dset_loaders["target"]) = 16

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_target.next()
            # inputs_test.size---->torch.Size([bs, 3, 224, 224])
            # _ 表示的是不可获得的label,共64个=batch-size
            # tar_idx 表示的是索引
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netG.eval()
            netF.eval()
            # get the pseudo labels
            mem_label1 = obtain_label(dset_loaders["test"], netG, netF, netC)
            mem_label2 = obtain_label_easy(dset_loaders["test"], netG, netD)
            mem_label1 = torch.from_numpy(mem_label1).cuda()
            mem_label2 = torch.from_numpy(mem_label2).cuda()
            # mem_label.size = torch.Size([795]) 由795个标签组成的张量。
            # high_label.size 不断更新，从400多到700多，伪标签逐渐准确。
            netG.train()
            netF.train()

        inputs_test = inputs_test.cuda()
        batch_size = inputs_test.shape[0]

        iter_num += 1
        lr_scheduler(optimizer_g, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_d, iter_num=iter_num, max_iter=max_iter)

        # Step A Train target date use CELoss
        total_loss1 = 0
        features_d = netG(inputs_test)
        features = netF(features_d)
        # features_test.shape ---->torch.Size([bs, 2048])
        outputs1 = netC(features)
        outputs2 = netD(features_d)
        # outputs_test.shape ---->torch.Size([bs, 31(num_class)])
        softmax_out1 = nn.Softmax(dim=1)(outputs1)
        softmax_out2 = nn.Softmax(dim=1)(outputs2)

        # loss of classifier discrepancy

        loss_skl = torch.mean(torch.sum(SKL(softmax_out1, softmax_out2), dim=1))

        total_loss1 += loss_skl * 0.1

        loss_ent = entropy(netD, features_d, args.lamda)

        total_loss1 += loss_ent

        optimizer_d.zero_grad()
        total_loss1.backward()
        optimizer_d.step()

        # Step B Train target date use Entropy
        for _ in range(1):
            total_loss2 = 0
            features_d = netG(inputs_test)
            features = netF(features_d)

            outputs1 = netC(features)
            outputs2 = netD(features_d)

            # outputs_test.shape ---->torch.Size([bs, 31(num_class)])
            softmax_out1 = nn.Softmax(dim=1)(outputs1)
            softmax_out2 = nn.Softmax(dim=1)(outputs2)

            # loss of refine label
            pred1 = mem_label1[tar_idx]
            pred2 = mem_label2[tar_idx]
            # pred.shape------>torch.Size([64(bs)])
            classifier_loss1 = nn.CrossEntropyLoss()(outputs1, pred1)
            classifier_loss2 = nn.CrossEntropyLoss()(outputs2, pred2)

            kl_distance = nn.KLDivLoss(reduction='none')
            log_sm = nn.LogSoftmax(dim=1)
            variance1 = torch.sum(kl_distance(log_sm(outputs1), softmax_out2), dim=1)
            variance2 = torch.sum(kl_distance(log_sm(outputs2), softmax_out1), dim=1)
            exp_variance1 = torch.mean(torch.exp(-variance1))  # 接近于1的数 0.9981，09683，，，，，
            exp_variance2 = torch.mean(torch.exp(-variance2))
            loss_seg1 = classifier_loss1 * exp_variance1 + torch.mean(variance1)
            loss_seg2 = classifier_loss2 * exp_variance2 + torch.mean(variance2)

            classifier_loss = args.alpha * loss_seg1 + (2 - args.alpha) * loss_seg2

            loss_cs = args.cls_par * classifier_loss

            total_loss2 += loss_cs

            # Loss of the entropy
            loss_ent1 = adentropy(netC, features, args.lamda)
            loss_ent2 = adentropy(netD, features_d, args.lamda)

            loss_mme = loss_ent1 + loss_ent2

            total_loss2 += loss_mme

            # loss of class balance
            loss_cb1 = class_balance(softmax_out1, args.lamda)
            loss_cb2 = class_balance(softmax_out2, args.lamda)

            loss_cb = loss_cb1 + loss_cb2

            total_loss2 += loss_cb

            if args.mix > 0:
                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs_test.size()[0]).cuda()
                mixed_input = lam * inputs_test + (1 - lam) * inputs_test[index, :]
                mixed_softout = (lam * softmax_out1 + (1 - lam) * softmax_out2[index, :]).detach()

                features_mix = netG(mixed_input)
                outputs_mixed1 = netC(netF(features_mix))
                outputs_mixed2 = netD(features_mix)

                outputs_mied_softmax1 = torch.nn.Softmax(dim=1)(outputs_mixed1)
                outputs_mied_softmax2 = torch.nn.Softmax(dim=1)(outputs_mixed2)

                loss_mix1 = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_mied_softmax1.log(), mixed_softout)
                loss_mix2 = args.mix * nn.KLDivLoss(reduction='batchmean')(outputs_mied_softmax2.log(), mixed_softout)

                loss_mix = loss_mix1 + loss_mix2

            total_loss2 += loss_mix

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss2.backward()
            optimizer_g.step()
            optimizer_d.step()

        # Test the accuracy
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netG.eval()
            netF.eval()
            acc1, acc_list1, accuracy1 = cal_acc(dset_loaders["test"], netG, netF, netC)
            acc2, acc_list2, accuracy2 = cal_acc_easy(dset_loaders["test"], netG, netD)
            acc_best1 = accuracy1
            acc_best2 = accuracy2
            log_str = (
                    "Task: {}, Iter:{}/{}; Accuracy_c = {:.2f}%, Accuracy_d = {:.2f}% ; Lcls : {:.6f}; Lent : {:.6f}".format(
                        args.name,
                        iter_num,
                        args.max_epoch * len(dset_loaders["target"]),
                        acc_best1,
                        acc_best2,
                        loss_cs.data,
                        loss_mme.data,
                    )
                    + "\n"
                    + str(acc_list1)
            )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")

            netG.train()
            netF.train()

        if args.savemodel:
            torch.save(netG.state_dict(), osp.join(args.output_dir, "target_G" + ".pt"))
            torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F" + ".pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C" + ".pt"))
            torch.save(netD.state_dict(), osp.join(args.output_dir, "target_D" + ".pt"))

    return netG, netF, netC, netD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCA on office-home")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=3, help="target")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")  # 128
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dset", type=str, default="officehome", choices=["office", "officehome", "visda"]
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--net", type=str, default="resnet50", help="resnet50, resnet101"
    )
    parser.add_argument('--lamda', type=float, default=0.45, metavar='LAM', help='value of lamda')  # 0.1
    parser.add_argument('--cls_par', type=float, default=0.15)  # 0.05
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2077, help="random seed")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")  # 1543
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="parameter1")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="dca")
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="double")
    parser.add_argument("--da", type=str, default="SFDA")
    parser.add_argument('--savemodel', type=bool, default=True)

    args = parser.parse_args()

    if args.dset == "office":
        names = ["amazon", "dslr", "webcam"]
        args.class_num = 31
    elif args.dset == "officehome":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65

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
    args.test_dset_path = args.t_dset_path

    current_folder = "./ckps/"
    args.output_dir_src = osp.join(
        current_folder, args.da, args.output, args.dset, names[args.s][0].upper()
    )  # # ./ckps/uda/bait/office/A
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

    args.out_file = open(
        osp.join(args.output_dir, "log_" + ".txt"),
        "w",
    )
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()
    train_target_mme(args)
