################################################################################
#
# LOGISTICS
#
#    Your name as in eLearning
#    Your UT Dallas identifier
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of ??.??% on epoch ?? with a learning
#    rate at that point of ?.????? and time required for each epoch of ~ ??? s
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept.
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#       For reference, an implementation of this network took ~ 112 s per epoch
#       for training, validation and checkpoint saving on Sep 27, 2020 using a
#       free GPU runtime in Google Colab.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import numpy as np
# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################
# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 512
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

#configurations
NUM_BLOCKS = [i for i in range(1, 17)]
BLOCK_WIDTH = [i for i in range(1, 1025) if i % 8 == 0]
BOTTLENECK_RATIO = [1, 2, 4]
GROUP_WIDTH = [1, 2, 4, 8, 16, 32]
SE_RARIO = 4



# model
# add model parameters here

# training
# add training parameters here

# file
# add file parameters here

################################################################################
#
# DATA
#
################################################################################
"""
# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

"""

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)


# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)





def run():
    torch.multiprocessing.freeze_support()


class Head(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Head, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(num_channels, num_classes)
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Stem(nn.Module):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x


class XBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio=None):
        super(XBlock, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        if se_ratio is not None:
            se_channels = in_channels // se_ratio
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Conv2d(inter_channels, se_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(se_channels, inter_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.se = None
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
        self.rl = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        if self.se is not None:
            x1 = x1 * self.se(x1)
        x1 = self.conv_block_3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.rl(x1 + x2)
        return x



class Stage(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio):
        super(Stage, self).__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0",XBlock(in_channels, out_channels, bottleneck_ratio, group_width, stride, se_ratio))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i),XBlock(out_channels, out_channels, bottleneck_ratio, group_width, 1, se_ratio))
    def forward(self, x):
        x = self.blocks(x)
        return x


class AnyNetX(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetX, self).__init__()
        # For each stage, at each layer, number of channels (block width / bottleneck ratio) must be divisible by group width
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", Stem(prev_block_width))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in enumerate(zip(ls_num_blocks, ls_block_width,
                                                                                         ls_bottleneck_ratio,
                                                                                         ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width, bottleneck_ratio, group_width, stride, se_ratio))
            prev_block_width = block_width
        self.net.add_module("head", Head(ls_block_width[-1], DATA_NUM_CLASSES))
        self.initialize_weight()

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.net(x)
        return x


class AnyNetXb(AnyNetX):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXb, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_bottleneck_ratio)) == 1


class AnyNetXc(AnyNetXb):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXc, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert len(set(ls_group_width)) == 1


class AnyNetXd(AnyNetXc):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXd, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        assert all(i <= j for i, j in zip(ls_block_width, ls_block_width[1:])) is True


class AnyNetXe(AnyNetXd):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio):
        super(AnyNetXe, self).__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width, stride, se_ratio)
        if len(ls_num_blocks > 2):
            assert all(i <= j for i, j in zip(ls_num_blocks[:-2], ls_num_blocks[1:-1])) is True


class RegNetX(AnyNetXe):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride,
                 se_ratio=None):
        # We need to derive block width and number of blocks from initial parameters.
        parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
        # At this points, for each stage, the above-calculated block width could be incompatible to group width
        # due to bottleneck ratio. Hence, we need to adjust the formers.
        # Group width could be swapped to number of groups, since their multiplication is block width
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        # print (ls_num_blocks)
        # print (ls_block_width)
        # print (ls_bottleneck_ratio)
        # print (ls_group_width)
        super(RegNetX, self).__init__(ls_num_blocks, ls_block_width.astype(np.int).tolist(), ls_bottleneck_ratio,ls_group_width.tolist(), stride, se_ratio)


class RegNetY(RegNetX):
    # RegNetY = RegNetX + SE
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride,
                 se_ratio):
        super(RegNetY, self).__init__(initial_width, slope, quantized_param, network_depth, bottleneck_ratio,
                                      group_width, stride, se_ratio)



def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(train_loader),[batch_time, data_time, losses, top1, top5],prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    num_iter_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        writer.add_scalar('Train/Loss', losses.avg, epoch * num_iter_per_epoch + i)
        writer.add_scalar('Train/Top1_acc', top1.avg, epoch * num_iter_per_epoch + i)
        writer.add_scalar('Train/Top5_acc', top5.avg, epoch * num_iter_per_epoch + i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader),[batch_time, losses, top1, top5],prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
              .format(top1=top1, top5=top5))
        writer.add_scalar('Test/Loss', losses.avg, epoch)
        writer.add_scalar('Test/Top1_acc', top1.avg, epoch)
        writer.add_scalar('Test/Top5_acc', top5.avg, epoch)

    return top1.avg


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)




from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

bottleneck_ratio=1
group_width=8
initial_width=24
slope=36
quantized_param=2.5
network_depth=13
stride=2
se_ratio=4
lr=0.1
momentum=0.9
weight_decay=5e-4
epochs=100
batch_size=512
log_path="tensorboard/signatrix_regnet_imagenet"
saved_path="trained_models"


writer = SummaryWriter(log_path)
import shutil

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, saved_path, filename="checkpoint.pth.tar"):
    file_path = os.path.join(saved_path, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(saved_path, "best_checkpoint.pth.tar"))



if __name__ == '__main__':
    run()
    model = RegNetY(initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width, stride, se_ratio)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay, nesterov=True)
    best_acc1 = 0
    model.train()
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train(dataloader_train, model, criterion, optimizer, epoch, writer)
        acc1 = validate(dataloader_test, model, criterion, epoch, writer)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        }, is_best, saved_path)





