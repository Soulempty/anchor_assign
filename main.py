import numpy as np
from utils import AnchorGenerator,bbox_overlaps
import xml.etree.ElementTree as ET
from collections import Counter
import math
import os
import random
import cv2

'''
本代码主要功能：
1、用于可视化预置anchor在图像上的分布情况，包括不同尺度、比例（根据设定的不同特征感受）
2、根据设定的iou正样本匹配阈值，可视化anchor召回标注框的情况，同时输出匹配数量
3、可视化批量图像的anchor与标注框的匹配情况
'''

class VisAncbhor(object):
    def __init__(self,strides=[8],scales=[2,3,4],ratios=[0.5,1.0,2.0],pos_iou_thr=0.6,min_pos_iou=0.3,pre_anchors=None,size=1536,match_low_quality=True,use_gtpos_iou=False):
        super(VisAncbhor,self).__init__()

        self.strides = strides
        self.scales = scales
        self.ratios = ratios
        self.pos_iou_thr = pos_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.size = size
        self.use_gtpos_iou = use_gtpos_iou
        self.anchor_generator = AnchorGenerator(strides,ratios,scales,pre_anchors)

        self.colors = [random.sample(range(1,255),3) for _ in range(30)] 
    def parse_xml(self,xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        #resize box based on img.
        r = 1.0
        dw, dh = 0, 0
        if self.size:
            wh = root.find('size')
            w = int(wh.find('width').text)
            h = int(wh.find('height').text)
            r = min(self.size/w,self.size/h)
            new_unpad = int(round(w * r)), int(round(h * r))
            dw,dh = self.size - new_unpad[0], self.size - new_unpad[1]
            dw, dh = np.mod(dw, 32)/2, np.mod(dh, 32)/2
        for obj in root.findall('object'):

            bnd_box = obj.find('bndbox')
            bbox = [
               int(round(float(bnd_box.find('xmin').text)*r+dw)),
               int(round(float(bnd_box.find('ymin').text)*r+dh)),
               int(round(float(bnd_box.find('xmax').text)*r+dw)),
               int(round(float(bnd_box.find('ymax').text)*r+dh))
            ]
            bboxes.append(bbox)
        return np.array(bboxes)

    def anchor_assign(self,bboxes1,bboxes2,Assign_nums,color_nums,mode='iou',pos_iou_thr=None,min_pos_iou=None):
        #bboxes1 (array): shape (m, 4)
        #bboxes2 (array): shape (n, 4)
        color_inds = np.zeros((bboxes2.shape[0], ))  
        inds = np.arange(bboxes2.shape[0])
        color_inds[inds] = inds % color_nums
        
        if bboxes1.shape[0] == 0:
            return bboxes2, color_inds
        overlaps = bbox_overlaps(bboxes1,bboxes2,mode)
        assigned_gt_inds = np.zeros((bboxes2.shape[0], ))
        gt_assign_nums = []
        
        max_overlaps, argmax_overlaps = overlaps.max(axis=0),overlaps.argmax(axis=0)# one anchor to one best gt(one gt to more anchor)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(axis=1),overlaps.argmax(axis=1)#one gt to a best anchor
        
        pos_iou_thr = pos_iou_thr if pos_iou_thr else self.pos_iou_thr
        min_pos_iou = min_pos_iou if min_pos_iou else self.min_pos_iou
        if self.use_gtpos_iou:
            gt_pos_iou = np.log2(((bboxes1[:, 2] - bboxes1[:, 0])*(bboxes1[:, 3] - bboxes1[:, 1]))**0.5).astype(np.int32)*self.pos_iou_thr/10
            pos_iou_thr = gt_pos_iou[argmax_overlaps]
            
        pos_inds = max_overlaps >= pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            for i in range(bboxes1.shape[0]):
                if gt_max_overlaps[i] >= min_pos_iou:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
        assigned_inds = assigned_gt_inds>0
        assigned_anchors = bboxes2[assigned_inds]
        color_inds = color_inds[assigned_inds]
        assign_nums = dict(Counter(assigned_gt_inds))
        for k in Assign_nums.keys():
            Assign_nums[k] += assign_nums.get(k,0)
        return assigned_anchors,color_inds

    def vis_single(self,img,xml_path=None,loc=[]):
        h,w = img.shape[:2]
        feat_sizes = [(h//s,w//s) for s in self.strides]
        anchors = self.anchor_generator.grid_anchors(feat_sizes,loc)
        gt_boxes = np.zeros((0,4))
        Assign_nums = {0:0}
        if xml_path:
            gt_boxes = self.parse_xml(xml_path)
            Assign_nums = {k+1:0 for k in range(len(gt_boxes))}
            #如果提供标注文件，先画出标注框，颜色为绿色
            for i in range(len(gt_boxes)):
                xmin,ymin,xmax,ymax = gt_boxes[i]
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),2)

        color_nums = len(self.scales)*len(self.ratios)
        
        #画出匹配的anchors，边框颜色按每个位置框的个数，从颜色表中顺序选取
        anchors = np.concatenate(anchors,axis=0)
        #for anc in anchors:
        assigned_anchors,color_inds = self.anchor_assign(gt_boxes,anchors,Assign_nums,color_nums)
                
        for i in range(len(assigned_anchors)):
            xmin,ymin,xmax,ymax = [int(x) for x in assigned_anchors[i]]
            print(xmax-xmin,ymax-ymin)
            ind = int(color_inds[i])
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),self.colors[ind],1)
        print("Assign_nums:",Assign_nums)
        return img, Assign_nums

    def vis(self,path,loc=[],use_xml=False,show=True):
        if os.path.isfile(path):
            img = cv2.imread(path)
            h,w = img.shape[:2]
            r = 1.0
            if self.size:
                r = min(self.size/w,self.size/h)
                new_unpad = int(round(w * r)), int(round(h * r))
                
                dw,dh = self.size - new_unpad[0], self.size - new_unpad[1]
                dw, dh = np.mod(dw, 32)/2, np.mod(dh, 32)/2
                print(dw,dh)
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114)) 
                print(img.shape)
            img_name = os.path.basename(path)
            xml_path = None if not use_xml else path.replace(img_name.split('.')[-1],'xml')
            res, Assign_nums = self.vis_single(img,xml_path,loc)
            if show:
                cv2.namedWindow("AnchorImage",0)
                cv2.resizeWindow("AnchorImage", 1200,1600)
                cv2.imshow("AnchorImage",res)
                cv2.waitKey(0)
            else:
                save_path = os.path.join(os.getcwd(),'anchorImgs')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+img_name,res)

        elif os.path.isdir(path):
            img_dir = path+'/source'
            fs = os.listdir(img_dir)
            save_path = os.path.join(path,'anchorImgs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for f in fs:
                img_path = os.path.join(img_dir,f)
                img = cv2.imread(img_path)
                h,w = img.shape[:2]
                r = 1.0
                if self.size:
                    r = min(self.size/w,self.size/h)
                    new_unpad = int(round(w * r)), int(round(h * r))
                    dw,dh = self.size - new_unpad[0], self.size - new_unpad[1]
                    dw, dh = np.mod(dw, 32)/2, np.mod(dh, 32)/2
                    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
                    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114)) 
                xml_path = None if not use_xml else os.path.join(path,'label',f.replace(f.split('.')[-1],'xml'))
                res, Assign_nums = self.vis_single(img,xml_path,loc)
                print("Gt Boxes' assigned anchors nums: ",Assign_nums)
                cv2.imwrite(save_path+'/'+f,res)
            print("Finish the visualization of all the images in the <%s>" % path)
            
    

if __name__ == "__main__":
    paths = ['27.jpg',"502.jpg",'528.jpg','951.jpg','1674.jpg','1806.jpg','6653.jpg','8725.jpg']
    use_gtpos_iou = False
    pre_anchors = [[[12,12],[17,19],[25,26],[36,38],[29,75],[56,55]],
                   [[78,79],[52,171],[190,69],[116,118],[172,167],[107,296]],
                   [[393,129],[251,240],[427,441],[543,729],[669,642],[887,795]]]
    anc = VisAncbhor(strides=[4,8,16,32,64],scales=[[4,8],[8,16],[16,24],[16,20],[12,16]],ratios=[0.5,1,2],pre_anchors=None,pos_iou_thr=0.7,min_pos_iou=0.2,match_low_quality=True,use_gtpos_iou=use_gtpos_iou)
    #在指定图像坐标处画出所有anchors，显示不同颜色，先尺度，后比例. show=False时，会自动在当前目录创建文件夹anchorImgs，并将结果保存到其中。
    #anc.vis(path,loc=[(811,718)],show=True)  
    #在图像所有位置画出所有anchors,显示结果比较密集
    #anc.vis(path,show=True)
    #设定use_xml=True，会将标注框与所有anchors进行匹配，将匹配结果与标注框进行显示；图像与xml标注文件需要在同一目录下。
    #anc.vis(path,use_xml=True,show=False)
    #指定文件夹(包括图像目录：sources,标注目录：label)，测试所有图像，并将结果保存到指定路径的上一级目录的anchorImgs文件夹下
    #path = "images"
    for p in paths:
        anc.vis(p,use_xml=True,show=True)
    # 较大目标使用pos_iou=0.6,较小目标使用pos_iou=0.5 很小目标是用0.4