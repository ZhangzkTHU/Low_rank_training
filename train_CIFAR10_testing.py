# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# import pandas as pd
import csv
import time
import logging

from models import *
from utils import progress_bar
from randomaug import RandAugment
# from models.vit import ViT
# from models.convmixer import ConvMixer


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--wandb', default=False, action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit_base')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
# parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--log-dir', default='logs_4', type=str, help='log save path')
parser.add_argument('--exp-name', default=None, type=str, help='experiment name')
# parser.add_argument('--save-dir', default='checkpoint_new', type=str, help='checkpoint save path')
parser.add_argument('--resume-path', default=None, type=str, help='checkpoint resume path')
parser.add_argument('--save-freq', default=50, type=int, help='checkpoint save frequency')
parser.add_argument('--depth', default=6, type=int, help='depth of vit')
parser.add_argument('--preempt', default=False, action='store_true', help='preempt mode')
# parser.add_argument('--finetune_epochs', default=20, type=int, help='evaluation frequency')
# parser.add_argument('--lr-scheduler', default='cosine', type=str, help='lr scheduler')

# params for factorization
parser.add_argument('--r', default=64, type=int, help='factorization rank')
parser.add_argument('--ff-layer', default=None, type=int, help='which ff to be factorized')
parser.add_argument('--attn-layer', default=None, type=int, help='which attn to be factorized')
parser.add_argument('--deep', default=False, action='store_true', help='use deep factorization')
parser.add_argument('--merge-freq', default=10, type=int, help='merge and reinit frequency')
parser.add_argument('--correct-coef-A', default=0.1, type=float, help='correct coef for A matrix')
parser.add_argument('--correct-coef-B', default=0.1, type=float, help='correct coef for B matrix')
parser.add_argument('--correct-scheduling', default=None, type=str, help='correct coef scheduling')
parser.add_argument('--reset-optimizer', default=False, action='store_true', help='reset optimizer')



args = parser.parse_args()



# random seed
# torch.manual_seed(2)


######################################### setup #########################################
# take in args
usewandb = args.wandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# create dir
print('==> Creating directory..')
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if args.preempt:
    args.exp_name = "preempt_" + str(args.exp_name) + f'_{args.net}_lr{args.lr}_bs{args.bs}'
else:
    args.exp_name = time.strftime("%Y-%-m-%d %H:%M") + '_' + str(args.exp_name) + f'_{args.net}_lr{args.lr}_bs{args.bs}'
exp_dir = os.path.join(args.log_dir, args.exp_name)
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
# print(f'{exp_dir}/{args.net}-{args.patch}-ckpt-best.pth')
# exit()


# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# factorziation settings

# Model factory..
print('==> Building model..')
if args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.relora_grad_test import ReLoRaModel
    from models.simplevit import SimpleViT
    net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = 128,
        depth = 4,
        heads = 4,
        mlp_dim = 128,
)
    # copy net to net_relora
    import copy
    net_relora = copy.deepcopy(net)
    net_relora = ReLoRaModel(
        model=net_relora,
        target_modules=["transformer"],
        r=args.r,
        lora_dropout=0,
        lora_alpha=args.r,
)
# elif args.net=="simplevit":
#     from models.simplevit import SimpleViT
#     net = SimpleViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 4,
#     heads = 8,
#     mlp_dim = 512
# )
# elif args.net=="simplevit_factorized":
#     from models.simplevit_factorized import SimpleViT_Factorized
#     net = SimpleViT_Factorized(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 4,
#     heads = 8,
#     mlp_dim = 512,
#     r=args.r
# )
# elif args.net=="simplevit_orthinit":
#     from models.SimpleViT_orthinit import SimpleViT
#     net = SimpleViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = args.depth,
#     heads = 8,
#     mlp_dim = 512
# )
# elif args.net=="my_vit":
#     from models.vit_factorized import ViT
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 3072,
# )
# elif args.net=="my_vit_factorized":
#     from models.vit_factorized import ViT_factorized
#     net = ViT_factorized(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 3072,
#     rs = [None] * 5 + [40]
# )
# elif args.net=="my_vit_overparameterized":
#     from models.vit_factorized import ViT_overparametrized
#     net = ViT_overparametrized(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 3072,
# )
elif args.net=="vit_base":
    from models.vit import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="mlpmixer_factorized":
    from models.mlpmixer import MLPMixer_factorized
    net = MLPMixer_factorized(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10,
    rs_patch = rs_patch,
    rs_channel = rs_channel,
    deep = args.deep
)
elif args.net=="simplevit":
    from models.relora_grad_test import ReLoRaModel
    from models.simplevit import SimpleViT
    net = ReLoRaModel(
        model=SimpleViT(image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 8,
        mlp_dim = 512),
        target_modules=["transformer"],
        r=args.r,
        lora_dropout=0,
    )


# elif args.net=="vit_base_factorizeAttn":
#     from models.vit_factorized import ViT_factorized
#     net = SimpleViT_FactorizeAttn(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 10,
#     dim = 768,
#     depth = 12,
#     heads = 12,
#     mlp_dim = 3072,
#     factorize_layer = [10 for _ in range(12)]
# )
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))


# For Multi-GPU
# if 'cuda' in device:
#     print(device)
#     print("using data parallel")
#     net = torch.nn.DataParallel(net) # make parallel
#     cudnn.benchmark = True

# if args.resume_path is not None:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir(args.resume_path), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(args.resume_path)
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
    
if 'cuda' in device:
    print(device)
    if torch.cuda.device_count() > 1:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        net_relora = torch.nn.DataParallel(net_relora) # make parallel
    else:
        net.to(device)
        net_relora.to(device)
    cudnn.benchmark = True

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    from adam_proj import Adam_proj
    optimizer_relora = optim.Adam(net_relora.parameters(), lr=args.lr)
    optimizer = Adam_proj(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Preempt mode
if args.preempt and os.path.exists(f'{exp_dir}/{args.net}-{args.patch}-ckpt-preempt.pth'):
    checkpoint = torch.load(f'{exp_dir}/{args.net}-{args.patch}-ckpt-preempt.pth', map_location=device)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch'] + 1
    print('Loading epoch{} model..'.format(start_epoch))


# save args and model architecture to a json
import json
with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
    n_params = sum(p.numel() for p in net.parameters())
    json.dump(vars(args)|{'n_gpus':torch.cuda.device_count(), 'n_params': n_params}, f, indent=4)
with open(os.path.join(exp_dir, 'architecture.txt'), 'w') as f:
    print(net, file=f)


##################################### Training #####################################

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net_relora.train()
    # if epoch % args.merge_freq == 0:
    #     print("Merging")
    #     net.merge()
    #     for batch_idx, (inputs, targets) in enumerate(trainloader):
    #         # print(f'batch: {batch_idx}')
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         loss = criterion(net(inputs), targets)
    #         loss.backward()
    #         # break
    #     print("Reinitializing")
    #     net.reinit()
    #     optimizer.zero_grad()

    train_loss = 0
    correct = 0
    total = 0
    train_loss_relora = 0
    correct_relora = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        # with torch.cuda.amp.autocast(enabled=use_amp):
        if batch_idx % args.merge_freq == 0:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # galore
            proj_dict = []
            print('Calculating projection matrix..')
            for param in net.parameters():
                if param.grad is not None:
                    if param.size() in [torch.Size([128, 256]), torch.Size([768, 128]), torch.Size([128, 128])]:
                        grad = param.grad
                        if grad.isnan().any():
                            print(grad)
                            exit()
                        _, _, VT = torch.linalg.svd(grad)
                        proj = VT.T[:, :args.r]
                        # use random orthogonal projection
                        # proj = torch.linalg.qr(torch.randn(grad.shape).T)[0]
                        # proj = proj.T[:args.r, :].to(device)
                        proj_dict.append(proj)
            optimizer.zero_grad()
            
            # net_relora
            print("Merging")
            net_relora.merge()
            outputs_relora = net_relora(inputs)
            loss_relora = criterion(outputs_relora, targets)
            loss_relora.backward()
            for name, param in net_relora.named_parameters():
                if param.size() in [torch.Size([768, 128])]:
                    fullgrad_relora = param.grad.data
                    break
            print("Reinitializing")
            # net_relora.reinit(proj_dict)
            net_relora.reinit()
            optimizer_relora.zero_grad()
             # zero out the momentum buffers
            if args.reset_optimizer:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if 'momentum_buffer' in state:
                            state['momentum_buffer'] = torch.zeros_like(p)


        outputs_relora = net_relora(inputs)
        loss_relora = criterion(outputs_relora, targets)
        loss_relora.backward()
        # print the gradient of the first layer
        for name, param in net_relora.named_parameters():
            if 'lora_B' in name and 'weight' in name:
                grad_relora_B = param.grad.data
                break
        for name, param in net_relora.named_parameters():
            if 'lora_A' in name and 'weight' in name:
                relora_A = param.data
                # print(relora_A.shape)
                break
        


        optimizer_relora.step()
        optimizer_relora.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for _, param in net.named_parameters():
            if param.size() in [torch.Size([768, 128])]:
                grad_galore = param.grad.data
                break
        optimizer.step(projs=proj_dict)
        optimizer.zero_grad()


        # print("relora full grad", fullgrad_relora[:10, :1], "galore full grad", grad_galore[:10, :1])
        print("full grad diff:", torch.norm((fullgrad_relora - grad_galore))/torch.norm(fullgrad_relora))
        # print("relora grad:", grad_relora_B[:10, :1], "galore grad:", (grad_galore @ relora_A.T)[:10, :1])
        print("update diff:", torch.norm((grad_relora_B - grad_galore@relora_A.T))/torch.norm(grad_relora_B))

        # print((relora_A - proj_dict[0]).norm())
        # assert (relora_A - proj_dict[0]).norm() < 1e-5
        # print()
        # import pdb
        # pdb.set_trace()

        # calculate correction coefficient with scheduling
        # if args.correct_scheduling == "linear":
        #     correct_coef_A1 = args.correct_coef_A * (1 - (batch_idx + len(trainloader)* epoch) / (len(trainloader)*args.merge_freq))
        #     correct_coef_B1 = args.correct_coef_B * (1 - (batch_idx + len(trainloader)* epoch) / (len(trainloader)*args.merge_freq))
        # elif args.correct_scheduling == "cosine":
        #     correct_coef_A1 = args.correct_coef_A * (1 + np.cos(np.pi * (batch_idx + len(trainloader) * epoch) / (len(trainloader)*args.merge_freq))) / 2
        #     correct_coef_B1 = args.correct_coef_B * (1 + np.cos(np.pi * (batch_idx + len(trainloader) * epoch) / (len(trainloader)*args.merge_freq))) / 2
        # else:
        #     correct_coef_A1 = args.correct_coef_A
        #     correct_coef_B1 = args.correct_coef_B

        # print(f'correct_coef_A1: {correct_coef_A1}, correct_coef_B1: {correct_coef_B1}')
        # print gradient before correction
        # for name, param in net.named_parameters():
        #     if 'lora_A' in name and 'weight' in name:
        #         oldweight_A = param.data
        #         oldgrad_A = param.grad.data
        #         break
        # for name, param in net.named_parameters():
        #     if 'lora_B' in name and 'weight' in name:
        #         oldgrad_B = param.grad.data
        #         break
        # print('layer1 A old weight scale:', torch.norm(oldweight_A))
        # print('layer1 A old grad scale:', torch.norm(oldgrad_A))
        # print('layer1 A old grad:', oldgrad_A[:5, :5])
        # print('layer1 B old grad:', oldgrad_B[:5, :5])
        # net.correct_grad(correct_coef_A1, 1-correct_coef_A1, correct_coef_B1, 1-correct_coef_B1)
        # print gradient after correction
        # for name, param in net.named_parameters():
        #     if 'lora_A' in name and 'weight' in name:
        #         newgrad_A = param.grad.data
        #         break
        # for name, param in net.named_parameters():
        #     if 'lora_B' in name and 'weight' in name:
        #         newgrad_B = param.grad.data
        #         break
        # print('layer1 A new grad:', newgrad_A[:5, :5])
        # print('layer1 B new grad:', newgrad_B[:5, :5])
        # calculate difference between old and new gradients
        # correction_A = (newgrad_A - oldgrad_A * (1-correct_coef_A1))/correct_coef_A1
        # print('layer1 A grad scale:', oldgrad_A.norm())
        # print('layer1 A correction scale:', correction_A.norm())
        # cossim_A = torch.nn.functional.cosine_similarity(oldgrad_A.flatten(), newgrad_A.flatten(), dim=0)
        # cossim_B = torch.nn.functional.cosine_similarity(oldgrad_B.flatten(), newgrad_B.flatten(), dim=0)
        # content = f'Step {batch_idx}, A grad scale: {float(oldgrad_A.norm())}, A correction scale: {float(correction_A.norm())}, A cossim: {float(cossim_A)}, B cossim: {float(cossim_B)}'

        # with open(os.path.join(exp_dir, 'graddiff.txt'), 'a') as appender:
        #     appender.write(content + "\n")
       
        # print(f'cossim_A: {cossim_A}, cossim_B: {cossim_B}')
        # import pdb
        # pdb.set_trace()



        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_relora += loss_relora.item()
        _, predicted_relora = outputs_relora.max(1)
        # total += targets.size(0)
        correct_relora += predicted_relora.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        print(f'Epoch {epoch}, batch {batch_idx}, train loss: {train_loss/(batch_idx+1):.5f}, acc: {(100.*correct/total):.5f}')
        print(f'Epoch {epoch}, batch {batch_idx}, train loss_relora: {train_loss_relora/(batch_idx+1):.5f}, acc_relora: {(100.*correct_relora/total):.5f}')
        
        import pdb
        pdb.set_trace()
    # if (epoch+1) % args.merge_freq == 0:
    #     print("Merging and Reinitializing")
    #     net.merge_and_reinit()
    
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss/(batch_idx+1):.5f}, acc: {(100.*correct/total):.5f}'
    print(content)
    with open(os.path.join(exp_dir, 'train.txt'), 'a') as appender:
        appender.write(content + "\n")
        

    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    # if acc > best_acc:
    #     print('Saving best model..')
    #     state = {"model": net.state_dict(),
    #           "optimizer": optimizer.state_dict(),
    #           "scaler": scaler.state_dict()}
    #     # if not os.path.isdir(args.checkpoint_dir):
    #     #     os.mkdir(args.checkpoint_dir)
    #     torch.save(state, f'{exp_dir}/{args.net}-{args.patch}-ckpt-best.pth')
    #     best_acc = acc
    
    if epoch % args.save_freq == 0:
        print('Saving epoch{} model..'.format(epoch))
        state = {"model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,}
        torch.save(state, f'{exp_dir}/{args.net}-{args.patch}-ckpt-epoch{epoch}.pth')
    
    if args.preempt:
        print('Saving for preempt')
        state = {"model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,}
        torch.save(state, f'{exp_dir}/{args.net}-{args.patch}-ckpt-preempt.pth')
    
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(os.path.join(exp_dir, 'test.txt'), 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    # if args.net != 'vit_timm':
    scheduler.step() # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(os.path.join(exp_dir, f'{args.net}_patch{args.patch}.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

    # if args.net == 'vit_timm' and epoch >= args.finetune_epochs-1:
    #     break

# log best acc
with open(os.path.join(exp_dir, 'test.txt'), 'a') as appender:
    appender.write(str(best_acc))
# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))