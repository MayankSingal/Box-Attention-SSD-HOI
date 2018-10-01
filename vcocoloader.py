import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import vsrl_utils as vu
from pycocotools.coco import COCO
from utils.augmentations import SSDAugmentation


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
        
                
                
#populateTrainDict()

class vcocoLoader(data.Dataset):

    def __init__(self, transform=None):
        from pycocotools.coco import COCO
        self.trainList, self.classes, self.coco = populateTrainDict()
        self.transform = transform
        self.imgs_dir = '/home/user/data/mscoco/images/train2014'
        self.num_classes = len(self.classes)
        
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt
        
        
    def __len__(self):
        return len(self.trainList)
        
        
    def pull_item(self, index):
    
        anno = self.trainList[index]
        #print(anno)
        key = int(anno[0])
        img = self.coco.loadImgs(int(key))[0]['file_name']
        img = img.split("_")[-1]
        imgPath = self.imgs_dir + "/" + img
        img = cv2.imread(imgPath)
        h,w,_ = img.shape
        scale = np.array([w,h,w,h])
        
        
        switch = anno[-1]
        boxes = []
        attention_box = []
        
        
        ### Parsing Anno File
        if switch == 0:
            for subjects in anno[1]:
                tmp_box = list(np.array(subjects) / scale)
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
                tmp_box = list(np.array(final_obj[:4]) / scale)
                tmp_box.append(self.classes.index(final_obj[-1]) + 2)
                #print(self.classes.index(final_obj[-1]) + 2, "CLASS")
                boxes += [tmp_box]
                
            attention_box += [anno[1]]
            
        ### Creating attention-based image
        
        attention_image = np.zeros((h,w,3), dtype=np.float32)
        
        if switch == 0:
            attention_image[:,:,1] = 1.0
        else:
            attention_image[:,:,2] = 1.0
        #print("SWITCH IS:", switch)
            
        for attBox in attention_box:
            x,y,xx,yy = int(attBox[0]), int(attBox[1]), int(attBox[2]), int(attBox[3])
            
            attention_image[y:yy,x:xx,0] = 1.0
            
                
        combined_image = np.concatenate((img, attention_image),2)
        #print(boxes)
        
        if self.transform is not None:
            boxes = np.array(boxes)
            img, boxes, labels = self.transform(combined_image, boxes[:, :4], boxes[:, 4], switch)
            img = img[:,:,(2,1,0,3,4,5)]
            boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        

        for j in range(len(boxes)):
            boxes[j][-1] -= 1
        #print(boxes, "BOXES")
        
        return torch.from_numpy(img).permute(2, 0, 1), boxes, h, w
        #return img, boxes, h, w
        
        
# transform = SSDAugmentation()
# testObj = vcocoLoader(transform)

# tmp = testObj.pull_item(200)


# img = tmp[2]
# print(tmp[1])


# cv2.imshow("test", img[:,:,3])
# cv2.imshow("test2", img[:,:,:3])
# cv2.waitKey(0) & 0xff
        
        
    
            
