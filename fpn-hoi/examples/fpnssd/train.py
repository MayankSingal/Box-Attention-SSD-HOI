'''FPNSSD512 train on KITTI.'''
from __future__ import print_function

import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.fpnssd import FPNSSDBoxCoder

from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort

import numpy as np


parser = argparse.ArgumentParser(description='PyTorch FPNSSD Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/fpnssd/model/fpnssd512_20_trained.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./weights/Epoch16.pth', type=str, help='checkpoint path')
args = parser.parse_args()

# Data
print('==> Preparing dataset..')
img_size = 512
box_coder = FPNSSDBoxCoder()
def transform_train(img, boxes, labels, switch):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img_, boxes_, labels_ = random_crop(img, boxes, labels)

    if switch == 1:
        while((0 not in labels) or (len(boxes_) == 1 and labels_[0]==0)) :
            img_, boxes_, labels_ = random_crop(img, boxes, labels)
    
    img, boxes, labels = img_, boxes_, labels_

    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)

    boxes_tmp = list(boxes.clone().data.cpu().numpy())
    labels_tmp = list(labels.clone().data.cpu().numpy())
   

    att_box = []

    if switch == 1:
       

        new_boxes = []
        new_labels = []

        for ii in range(len(labels_tmp)):
            if labels_tmp[ii] != 0:
                new_boxes.append(boxes_tmp[ii])
                new_labels.append(labels_tmp[ii])
            else:
                att_box.append(boxes_tmp[ii])

        boxes = torch.from_numpy(np.array(new_boxes))
        labels = torch.from_numpy(np.array(new_labels))    
    

    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels, att_box #boxes_tmp, labels_tmp

trainset = ListDataset(root='/home/user/Mayank/box-attention/data/VOC/VOCdevkit/VOC2012/JPEGImages',    \
                       list_file='torchcv/datasets/voc/voc12_trainval.txt', \
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

testset = ListDataset(root='/home/user/Mayank/box-attention/data/VOC/VOCdevkit/VOC2012/JPEGImages',  \
                      list_file='torchcv/datasets/voc/voc12_test.txt', \
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=6, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=1)

# Model
print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = FPNSSD512(num_classes=31).to(device)
#net = torch.nn.DataParallel(net)
#net.load_state_dict(torch.load(args.model))
temp = torch.load(args.model)

new_state_dict = {}
for k,v in temp.items():
    if "extractor" in k:
        name = "fpn" + k[9:]
        new_state_dict[name] = v

net.load_state_dict(new_state_dict, strict = False)



if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint)
    #best_loss = checkpoint['loss']
    #start_epoch = checkpoint['epoch']

criterion = SSDLoss(num_classes=31)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, attention_image) in enumerate(trainloader):
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)
        attention_image = attention_image.to(device)


        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs, attention_image)

        # print(loc_preds)
        # print(loc_targets, "TARG")

       
        loss = criterion(loc_preds.float(), loc_targets.float(), cls_preds.float(), cls_targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.item(), train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

    torch.save(net.state_dict(), "/home/user/Mayank/misc/torchcv/weights/Epoch" + str(epoch) + ".pth")

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.item()
            print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.item(), test_loss/(batch_idx+1), batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    #test(epoch)
