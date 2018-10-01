from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import cv2
import numpy as np
import glob

labelmap = ['human','hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit_1', 'eat_1', 'jump', 'lay', 'talk_on_phone', 'carry', 'throw', 'catch', 'cut_1', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink', 'kick', 'point', 'read', 'snowboard', 'eat_2', 'cut_2', 'hit_2']
for ii,elem in enumerate(labelmap):
    print(elem, ii)


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_COCO_40000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def test_net(image_path, net, cuda, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img2 = img.copy()
    
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    h,w,_ = 300,300,3

    ## Add attention maps here
    attention_image = np.zeros((h,w,3), dtype=np.float32)
    attention_image[:,:,1] = 1.0
    attention_image = Variable(torch.from_numpy(attention_image)).permute(2,0,1).unsqueeze(0)

    x = torch.cat((x, attention_image), dim=1)
    ##########################

    if cuda:
        x = x.cuda()

    y = net(x)      # forward pass
    detections = y.data
    #print(detections.size())

    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    pred_num = 0
    subject_predictions = []
    for i in [1]:
        j = 0
        while detections[0, i, j, 0] >= 0.5:
               
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = [pt[0], pt[1], pt[2], pt[3]]
            subject_predictions.append(coords)
            #print(coords)
            cv2.rectangle(img, (pt[0],pt[1]), (pt[2],pt[3]), (255,0,0),2)

            j += 1

    print(subject_predictions)
    if len(subject_predictions) == 0:
        print("No subjects detected")
        return "No subjects detected"

    

    for attBox in subject_predictions:
        attention_image = np.zeros((h,w,3), dtype=np.float32)
        attention_image[:,:,2] = 1.0
        x,y,xx,yy = int(attBox[0]), int(attBox[1]), int(attBox[2]), int(attBox[3])    
        attention_image[y:yy,x:xx,0] = 1.0
        attention_image = Variable(torch.from_numpy(attention_image)).permute(2,0,1).unsqueeze(0)

        x = torch.from_numpy(transform(img2)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        x = torch.cat((x, attention_image), dim=1)
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        #print(detections.size())

        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        object_predictions = []
        for i in range(2,detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.11:
               
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                object_predictions.append(coords)
                
                cv2.rectangle(img, (pt[0],pt[1]), (pt[2],pt[3]), (255,255,0),2)
                cv2.putText(img, label_name, (pt[0], pt[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                j += 1

    img = cv2.resize(img, (600,600))
    cv2.imshow("test", img.copy())
    cv2.waitKey(0) & 0xff


def test_box_attention():
    # load net
    num_classes = 30 + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
  
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    for image_path in glob.glob("testImages/*"):
        test_net(image_path, net, args.cuda,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=0.1)

if __name__ == '__main__':
    test_box_attention()
