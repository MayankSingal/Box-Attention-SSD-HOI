from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import vsrl_utils as vu

from PIL import Image
import numpy as np
import cv2

def populateTrainDict():
    imgs_dir = '/home/user/data/mscoco/images/train2014'
    coco = vu.load_coco()
    vcoco_all = vu.load_vcoco('vcoco_trainval')

    classes = [x['action_name'] for x in vcoco_all]
    ## Modifying classes according to new criteria of obj-instr
    classes[classes.index("eat")] = "eat_obj"
    classes[classes.index("cut")] = "cut_obj"
    classes[classes.index("hit")] = "hit_obj"
    classes.append("eat_instr")
    classes.append("cut_instr")
    classes.append("hit_instr")
   
    print("# of VCOCO classes:", len(classes))
   


    for x in vcoco_all:
        x = vu.attach_gt_boxes(x,coco)
        
    data_dict = {}
    
    for idx in range(len(vcoco_all)):
    
        actDict = vcoco_all[idx]
        #print(actDict['role_name'], actDict['action_name'])
        
        #continue
        
        
        img_ids = [x[0] for x in actDict['image_id']]
        
        for i,ids in enumerate(img_ids):
            ids = int(ids)
            
            
            if actDict['label'][i][0] == 1:
                if ids in data_dict:
                    tmp = list(actDict['role_bbox'][i])                          
                    tmp.append(actDict['action_name'])
                    data_dict[ids].append(tmp)
                else:
                    data_dict[ids] = []
                    tmp = list(actDict['role_bbox'][i])                 
                    tmp.append(actDict['action_name'])
                    data_dict[ids].append(tmp)
        
    
    data_list = []
    for ids in data_dict.keys():
        img_anno = data_dict[ids]
        tmp_subs = []
        for elem in img_anno:
            tmp_subs.append(tuple(elem[:4]))
        subject_set = set(tmp_subs)
        
        subject_tmp = []
        for subj in subject_set:
            subject_tmp.append(list(subj))
        data_list.append([ids, subject_tmp, 0])
        
        tmpDict = {}
        for subj in subject_set:
            tmpDict[subj] = []
        
        for elem in img_anno:
            subj = tuple(elem[:4])
            tmpDict[subj].append(elem[4:])
        
        for subj in tmpDict.keys():
            
            data_list.append([ids, list(subj), tmpDict[subj], 1])

    for elem in data_list:
        if elem[-1] == 1:
            for i in range(len(elem[2])):
                if elem[2][i][-1] in ['cut', 'hit']:

                    tmp = elem[2][i][4:]
                    tmp[-1] += "_obj"
                    tmpLabel = elem[2][i][-1] + "_instr"
                    elem[2].append(tmp)


                    elem[2][i] = elem[2][i][:4]
                    elem[2][i].append(tmpLabel)
                elif elem[2][i][-1] in ['eat']:

                    tmp = elem[2][i][4:]
                    tmp[-1] += "_instr"
                    tmpLabel = elem[2][i][-1] + "_obj"
                    elem[2].append(tmp)


                    elem[2][i] = elem[2][i][:4]
                    elem[2][i].append(tmpLabel)

                 
    return data_list, classes, coco



class ListDatasetOld(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        

        boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()

        #print(labels)
        #print(boxes.size(),"BSIZE")

        
        

        if self.transform:
            img, boxes, labels, boxes_tmp = self.transform(img, boxes, labels, 0)

       
        #print(img.size())
        #print(boxes_tmp)
       
        return img, boxes, labels, np.ones((10,10))

    def __len__(self):
        return self.num_imgs


class ListDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.

    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        from pycocotools.coco import COCO
        self.trainList, self.classes, self.coco = populateTrainDict()
        self.transform = transform
        self.imgs_dir = '/home/user/data/mscoco/images/train2014'
        self.num_classes = len(self.classes)       
        

    def __getitem__(self, idx):
      
        img, boxes, labels, att_img  = self.pull_item(idx)

        return img, boxes, labels.long(), att_img

    def pull_item(self,index):

        anno = self.trainList[index]
        #print(anno)
        key = int(anno[0])
        img = self.coco.loadImgs(int(key))[0]['file_name']
        img = img.split("_")[-1]
        imgPath = self.imgs_dir + "/" + img
        img = Image.open(os.path.join(imgPath))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        switch = anno[-1]
        boxes = []


        if switch == 0:
            for subjects in anno[1]:
                tmp_box = list(np.array(subjects))
                tmp_box.append(1)
                boxes += [tmp_box]
            
        else:
            for objects in anno[2]:
                if (len(objects)==1) or (np.isnan(objects[0])):
                    final_obj = anno[1]
                    final_obj.append(objects[-1])
                else:
                    final_obj = objects
                
                #print(final_obj)
                tmp_box = list(np.array(final_obj[:4]))
                tmp_box.append(self.classes.index(final_obj[-1]) + 2)
                #print(self.classes.index(final_obj[-1]) + 2, "CLASS")
                boxes += [tmp_box]

            tmp_sub = anno[1][:4]
            tmp_sub.append(1)
            boxes += [tmp_sub]

        boxes = np.array(boxes)
        #print(boxes, "BOXES")

        labels = boxes[:,4] - 1
        #print(labels)
        boxes = boxes[:,:4]

        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).float()

        # Transform always applied
        img, boxes, labels, attention_boxes = self.transform(img, boxes, labels, switch)

        #tmpImg = img.permute(1,2,0).data.cpu().numpy()


        
        #Find attention Boxes
        
        attention_image = np.zeros((512,512,3), dtype=np.float32)   

        if switch == 0:
            attention_image[:,:,1] = 1.0
        else:
            attention_image[:,:,2] = 1.0
            for attBox in attention_boxes:
                x,y,xx,yy = int(attBox[0]), int(attBox[1]), int(attBox[2]), int(attBox[3])
                
                attention_image[y:yy,x:xx,0] = 1.0

        # cv2.imshow("test", tmpImg)
        # cv2.imshow("attMap", attention_image[:,:,0])
        # cv2.imshow("attMap2", attention_image[:,:,1])
        # cv2.imshow("attMap3", attention_image[:,:,2])
        # cv2.waitKey(0)

        


        return img, boxes, labels, torch.from_numpy(attention_image).permute(2,0,1)
               
           




        



        


















    def __len__(self):
        return len(self.trainList)