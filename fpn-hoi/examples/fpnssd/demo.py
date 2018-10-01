import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont
from torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder

import numpy as np
import glob
import cv2

labelmap = ['human','hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit_1', 'eat_1', 'jump', 'lay', 'talk_on_phone', 'carry', 'throw', 'catch', 'cut_1', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink', 'kick', 'point', 'read', 'snowboard', 'eat_2', 'cut_2', 'hit_2']
fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

#print('Loading model..')
net = FPNSSD512(num_classes=31).to('cuda')
#net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./weights/Epoch5.pth'))
#net.eval()


for image_path in glob.glob("testImages/*"):
	print('Loading image..')
	img = Image.open(image_path)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	ow = oh = 512
	img = img.resize((ow,oh))

	print('Predicting..')
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
	x = transform(img).cuda()
	x2 = x.clone()

	attention_image = np.zeros((ow,oh,3), dtype=np.float32)
	attention_image[:,:,1] = 1.0
	attention_image = torch.from_numpy(attention_image).permute(2,0,1).unsqueeze(0).cuda()


	loc_preds, cls_preds = net(x.unsqueeze(0), attention_image)

	print('Decoding..')
	box_coder = FPNSSDBoxCoder()
	loc_preds = loc_preds.squeeze().cpu()
	cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
	print(loc_preds.size())
	boxes, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh=0.6)

	for box in boxes:
		for j in range(len(box)):
			if box[j] < 0:
				box[j] = 0

	print(boxes)
	if len(boxes) == 0:
		continue

	draw = ImageDraw.Draw(img)
	for box in boxes:
	    draw.rectangle(list(box), outline='red')

	for attBox in boxes:
		attention_image = np.zeros((oh,ow,3), dtype=np.float32)
		attention_image[:,:,2] = 1.0
		x_,y,xx_,yy = int(attBox[0]), int(attBox[1]), int(attBox[2]), int(attBox[3])    
		attention_image[y:yy,x_:xx_,0] = 1.0
		attention_image = torch.from_numpy(attention_image).permute(2,0,1).unsqueeze(0).cuda()

		loc_preds, cls_preds = net(x2.unsqueeze(0), attention_image)

		box_coder = FPNSSDBoxCoder()
		loc_preds = loc_preds.squeeze().cpu()
		cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
		boxes_, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh = 0.09)
		for kk,box in enumerate(boxes_):
			draw.rectangle(list(box), outline='blue')
			draw.text((list(box)[0],list(box)[1]), labelmap[labels[kk]], font=fnt, fill=(255,255,255,128))


	img.show()
	cv2.waitKey(1000) & 0xff
	



