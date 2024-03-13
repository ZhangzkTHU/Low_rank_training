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
parser.add_argument('--depth', default=12, type=int, help='depth of vit')
parser.add_argument('--preempt', default=False, action='store_true', help='preempt mode')
# parser.add_argument('--finetune_epochs', default=20, type=int, help='evaluation frequency')
# parser.add_argument('--lr-scheduler', default='cosine', type=str, help='lr scheduler')

# params for factorization
parser.add_argument('--r', default=40, type=int, help='projection rank')
parser.add_argument('--project-freq', default=10, type=int, help='projection frequency')
parser.add_argument('--warmup-epochs', default=5, type=int, help='warmup epochs')

# parser.add_argument('--ff-layer', default=None, type=int, help='which ff to be factorized')
# parser.add_argument('--attn-layer', default=None, type=int, help='which attn to be factorized')
# parser.add_argument('--deep', default=False, action='store_true', help='use deep factorization')
# parser.add_argument('--patch-layer', default=None, type=int, help='which patch-mixer to be factorized')
# parser.add_argument('--channel-layer', default=None, type=int, help='which channel-mixer to be factorized')



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
    dropout = 0,
    emb_dropout = 0
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_base":
    from models.vit import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = 768,
    depth = args.depth,
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
    else:
        net.to(device)
    cudnn.benchmark = True

# Loss is CE
criterion = nn.CrossEntropyLoss()

# if args.opt == "adam":
#     optimizer = optim.Adam(net.parameters(), lr=args.lr)
# elif args.opt == "sgd":
#     optimizer = optim.SGD(net.parameters(), lr=args.lr)

from adam_proj import Adam_proj
# if args.opt == "adam_proj":
optimizer_0 = optim.Adam(net.parameters(), lr=args.lr)
optimizer = Adam_proj(net.parameters(), lr=args.lr)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_0, args.n_epochs)
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
stable_rank_all = np.zeros((12, args.warmup_epochs))
cutoff_rank_all = np.zeros((12, args.warmup_epochs))
nuclear_norm_all = np.zeros((12, args.warmup_epochs))
effective_rank_all = np.zeros((12, args.warmup_epochs))

def train(epoch):
    print('\nEpoch: %d' % epoch)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    svals_all = []
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        layer_idx = 0
        if batch_idx == 50:
            print('Calculating projection matrix..')
            for param in net.parameters():
                if param.grad is not None:
                    # print(param.size())
                    if param.size() in [torch.Size([3072, 768])]:
                        # if layer_idx in [0, 4, 8, 11]:
                        grad = param.grad.detach()
                        # print(param.size(), grad.size())
                        if grad.isnan().any():
                            print(grad)
                            exit()
                        svals = torch.linalg.svdvals(grad)
                        # print(svals[:args.r])

                        # compute stable rank
                        # print(svals[:40])
                        stable_rank = (svals**2).sum()/(svals[0]**2)
                        cutoff_rank = (svals/svals[0] >0.01).sum()
                        nuclear_norm = (svals/svals[0]).sum()
                        svals_normalized = svals/svals.sum()
                        effective_rank = torch.exp((-svals_normalized * torch.log(svals_normalized)).sum())

                        stable_rank_all[layer_idx][epoch] = int(stable_rank)
                        cutoff_rank_all[layer_idx][epoch] = int(cutoff_rank)
                        nuclear_norm_all[layer_idx][epoch] = int(nuclear_norm)
                        effective_rank_all[layer_idx][epoch] = int(effective_rank)

                        # print(f"stable rank: {stable_rank}, cutoff rank: {cutoff_rank}, nuclear norm: {nuclear_norm}, effective rank: {effective_rank}")
                    
                        layer_idx += 1
            
        optimizer_0.step()
        optimizer_0.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1), stable_rank_all, cutoff_rank_all, nuclear_norm_all, effective_rank_all
    

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
    
    if (epoch+1) % args.save_freq == 0:
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
    with open(os.path.join(exp_dir, 'accs.txt'), 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
# proj_dict = None

if usewandb:
    wandb.watch(net)

net.cuda()
svals_epochs = []
for epoch in range(start_epoch, args.warmup_epochs):
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

with open(os.path.join(exp_dir, f'stable_rank.txt'), 'w') as f:
    for layer_idx in range(12):
        f.write(f'\nlayer{layer_idx}\n')
        f.write(str(stable_rank_all[layer_idx]))
    f.close()
with open(os.path.join(exp_dir, f'cutoff_rank.txt'), 'w') as f:
    for layer_idx in range(12):
        f.write(f'\nlayer{layer_idx}\n')
        f.write(str(cutoff_rank_all[layer_idx]))
    f.close()
with open(os.path.join(exp_dir, f'nuclear_norm.txt'), 'w') as f:
    for layer_idx in range(12):
        f.write(f'\nlayer{layer_idx}\n')
        f.write(str(nuclear_norm_all[layer_idx]))
    f.close()
with open(os.path.join(exp_dir, f'effective_rank.txt'), 'w') as f:
    for layer_idx in range(12):
        f.write(f'\nlayer{layer_idx}\n')
        f.write(str(effective_rank_all[layer_idx]))
    f.close()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# plot stable rank of the 1, 4, 8, 12 layers
alphas = [0.3, 0.5, 0.7, 1.0]
for i, layer_idx in enumerate([0, 3, 7, 11]):
    axs[0, 0].plot(stable_rank_all[layer_idx], label=f'layer{layer_idx+1}', linewidth=2, color='red', alpha=alphas[i])

# plot cutoff rank of the 1, 4, 8, 12 layers
for i, layer_idx in enumerate([0, 3, 7, 11]):
    axs[0, 1].plot(cutoff_rank_all[layer_idx], label=f'layer{layer_idx+1}', linewidth=2, color='blue', alpha=alphas[i])

# plot nuclear norm of the 1, 4, 8, 12 layers
for i, layer_idx in enumerate([0, 3, 7, 11]):
    axs[1, 0].plot(nuclear_norm_all[layer_idx], label=f'layer{layer_idx+1}', linewidth=2, color='green', alpha=alphas[i])

# plot effective rank of the 1, 4, 8, 12 layers
for i, layer_idx in enumerate([0, 3, 7, 11]):
    axs[1, 1].plot(effective_rank_all[layer_idx], label=f'layer{layer_idx+1}', linewidth=2, color='purple', alpha=alphas[i])

# Set titles and labels
axs[0, 0].set_title('Stable Rank')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Rank')
axs[0, 0].legend(fontsize='large')

axs[0, 1].set_title('Cutoff Rank')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Rank')
axs[0, 1].legend(fontsize='large')

axs[1, 0].set_title('Nuclear Norm')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Norm')
axs[1, 0].legend(fontsize='large')

axs[1, 1].set_title('Effective Rank')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Rank')
axs[1, 1].legend(fontsize='large')

# Adjust spacing between subplots
plt.tight_layout()

# Save the combined plot
plt.savefig(os.path.join(exp_dir, 'combined_plots.png'))






# log best acc
with open(os.path.join(exp_dir, 'accs.txt'), 'a') as appender:
    appender.write(str(best_acc))
# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))