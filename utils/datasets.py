# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg
import cv2
import numpy as np
import random
import torchvision.transforms as transforms
import utils.data_augment as dataAug
import utils.tools as tools

def get_image_id(filename):
    return int(os.path.basename(filename).split(".")[0])

class Build_Train_Dataset(Dataset):
    def __init__(self, anno_file, anno_file_type, img_size=416):
        self.img_size = img_size  # For Multi-training
        if cfg.TRAIN.DATA_TYPE == 'VOC':
            self.classes = cfg.VOC_DATA.CLASSES
        elif cfg.TRAIN.DATA_TYPE == 'COCO':
            self.classes = cfg.COCO_DATA.CLASSES
        else:
            self.classes = cfg.DATASET.CLASSES
        self.cross_offset = 0.2
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file, anno_file_type)
        self.hue_jitter = 0.005
        self.bright_jitter = 0.25
        self.sat_jitter = 0.25
        self.label_smooth = dataAug.LabelSmooth()
        self.bbox_minsize = 40
        
    def __len__(self):
        return  len(self.__annotations)

    def __get_frag(self, mosaic_nr, cx, cy, img, bboxes):
        h, w, _ = img.shape
        
        if mosaic_nr == 0:
            width_of_nth_pic = cx 
            height_of_nth_pic = cy
        elif mosaic_nr == 1:
            width_of_nth_pic = self.img_size - cx
            height_of_nth_pic = cy
        elif mosaic_nr == 2:
            width_of_nth_pic = cx
            height_of_nth_pic = self.img_size - cy
        elif mosaic_nr == 3:
            width_of_nth_pic = self.img_size - cx
            height_of_nth_pic = self.img_size - cy
        # top left corner
        cut_x1 = random.randint(0, int(w * 0.33))
        cut_y1 = random.randint(0, int(h * 0.33))
          
        # Now we should find which axis should we randomly enlarge (this we do by finding out which ratio is bigger); cross x is basically width of the top left picture        
        if (w - cut_x1) / width_of_nth_pic < (h - cut_y1) / height_of_nth_pic:
            cut_x2 = random.randint(cut_x1 + int(w * 0.67), w)
            cut_y2 = int(cut_y1 + (cut_x2-cut_x1)/width_of_nth_pic*height_of_nth_pic)
        else:
            cut_y2 = random.randint(cut_y1 + int(h * 0.67), h)
            cut_x2 = int(cut_x1 + (cut_y2-cut_y1)/height_of_nth_pic*width_of_nth_pic)            
        
        img = cv2.resize(img[cut_y1:cut_y2, cut_x1:cut_x2, :],(width_of_nth_pic, height_of_nth_pic))
        
        w_ratio = width_of_nth_pic / (cut_x2 - cut_x1)
        h_ratio = height_of_nth_pic / (cut_y2 - cut_y1)

        # SHIFTING TO CUTTED IMG SO X1 Y1 WILL 0
        bboxes[:, [0,2]] -= cut_x1
        bboxes[:, [1,3]] -= cut_y1

        # RESIZING TO CUTTED IMG SO X2 WILL BE 1
        bboxes[:, [0,2]] *= w_ratio
        bboxes[:, [1,3]] *= h_ratio

        # CLAMPING BOUNDING BOXES, SO THEY DO NOT OVERCOME OUTSIDE THE IMAGE
        bboxes[:, [0,2]] = bboxes[:, [0,2]].clip(0, width_of_nth_pic-1)
        bboxes[:, [1,3]] = bboxes[:, [1,3]].clip(0, height_of_nth_pic-1)

        # RESIZING TO MOSAIC
        if mosaic_nr == 1:
            bboxes[:, [0,2]] += cx
        elif mosaic_nr == 2:
            bboxes[:, [1,3]] += cy
        elif mosaic_nr == 3:
            bboxes[:, [0,2]] += cx
            bboxes[:, [1,3]] += cy

        # FILTER TO THROUGH OUT ALL SMALL BBOXES
        filter_minbbox = (bboxes[:, 3] - bboxes[:, 1]) > self.bbox_minsize
        bboxes = bboxes[filter_minbbox]
        return img, bboxes        
        
    def __get_mosaic(self,idx):
        
        mosaic_img = np.zeros((self.img_size,self.img_size,3))
        
        cross_x = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))
        cross_y = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))
        
        raw_frag_img, raw_frag_bboxes = self.__parse_annotation(self.__annotations[idx])
        frag_img, frag_bboxes = self.__get_frag(0, cross_x, cross_y, raw_frag_img, raw_frag_bboxes)
        bboxes = frag_bboxes
        for i in range(1, 4):
            frag_idx = random.randint(0, len(self.__annotations)-1)
            raw_frag_img, raw_frag_bboxes = self.__parse_annotation(self.__annotations[frag_idx])
            frag_img, frag_bboxes = self.__get_frag(i, cross_x, cross_y, raw_frag_img, raw_frag_bboxes)
            if i==1:
                mosaic_img[0:cross_y, cross_x:, :] = frag_img
            elif i==2:
                mosaic_img[cross_y:, 0:cross_x, :] = frag_img
            elif i==3:
                mosaic_img[cross_y:, cross_x:, :] = frag_img
            bboxes = np.concatenate([bboxes, frag_bboxes])
        return mosaic_img, bboxes
    
    def __getitem__(self, item):
        assert item <= len(self), 'index range error'

        img, bboxes = self.__get_mosaic(item) if random.random() < 0.5 else self.__parse_annotation(self.__annotations[item])
        img = img.transpose(2, 0, 1)  # HWC->CHW 
        # print(bboxes)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def __load_annotations(self, anno_file, anno_type):

        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        with open(os.path.join(cfg.DATA_PATH,anno_file), 'r') as f:
            annotations = list(filter(lambda x:len(x)>0, f.readlines()))
        assert len(annotations)>0, "No images found in {}".format(anno_file)

        return annotations

    def __parse_annotation(self, annotation):
        """
        Data augument.
        :param annotation: Image' path and bboxes' coordinates, categories.
        ex. [image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...]
        :return: Return the enhanced image and bboxes. bbox'shape is [xmin, ymin, xmax, ymax, class_ind]
        """
        anno = annotation.strip().split(' ')

        img_path = os.path.join(cfg.DATA_PATH, anno[0])
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path

        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        
        img = img.astype(np.float32)
        
        # mosaic include mostly crop and moving. so no need for crop and affine transform here
        if random.random() < 0.5:
            img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2HSV).astype(np.float32)
            img[:,:,0] *= random.uniform(1-self.hue_jitter, 1+self.hue_jitter)
            img[:,:,1] *= random.uniform(1-self.sat_jitter, 1+self.sat_jitter)
            img[:,:,2] *= random.uniform(1-self.bright_jitter, 1+self.bright_jitter)
            img = cv2.cvtColor(np.uint8(img),cv2.COLOR_HSV2BGR)
        
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = img.shape[1] - bboxes[:, [2, 0]] - 1
            
        resize_ratio = min(1.0 * self.img_size / img.shape[1], 1.0 * self.img_size / img.shape[0])
        resize_w = int(resize_ratio * img.shape[0])
        resize_h = int(resize_ratio * img.shape[1])
        image_resized = cv2.resize(img, (resize_w, resize_h))

        img = np.full((self.img_size, self.img_size, 3), 128.0)
        dw = int((self.img_size - resize_w) / 2)
        dh = int((self.img_size - resize_h) / 2)
        img[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
        img = cv2.cvtColor(np.uint8(img),cv2.COLOR_BGR2RGB)
        
        return img/255.0 , bboxes.clip(0,self.img_size-1)
    def __creat_label(self, bboxes):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """

        anchors = np.array(cfg.MODEL.ANCHORS)
        strides = np.array(cfg.MODEL.STRIDES)
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL.ANCHORS_PER_SCLAE

        label = [np.zeros((int(train_output_size[i]), 
                           int(train_output_size[i]), 
                           anchors_per_scale, 
                           6+self.num_classes)) for i in range(3)]
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((90, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))
        # print(bboxes)
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = self.label_smooth(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("bbox_xywh: ", bbox_xywh)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = 1.0
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 90)  # BUG : 90为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = best_anchor_ind // anchors_per_scale
                best_anchor = best_anchor_ind % anchors_per_scale

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = 1.0
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 90)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

class Build_VAL_Dataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        truth = {}
        f = open(os.path.join(cfg.DATA_PATH, cfg.VAL.ANNO_FILE), 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.rstrip().split(" ")
            truth[data[0]] = []
            if len(data) > 1:
                for i in data[1:]:
                    truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())
        
    def get_image_id(filename):
        return int(os.path.basename(filename).split(".")[0])
    
    def __len__(self):
        return len(self.truth.keys())
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        
        img = cv2.imread(os.path.join(cfg.DATA_PATH, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        if num_objs > 0:
            boxes = bboxes_with_cls_id[...,:4]
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
            target['image_id'] = torch.tensor([get_image_id(img_path)])
            target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
            target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            target['boxes'] = torch.as_tensor([], dtype=torch.float32)
            target['labels'] = torch.as_tensor([], dtype=torch.int64)
            target['image_id'] = torch.tensor([get_image_id(img_path)])
            target['area'] = torch.as_tensor([], dtype=torch.float32)
            target['iscrowd'] = torch.as_tensor([], dtype=torch.int64)
            
        return img, target

if __name__ == "__main__":
    import sys
    anno_file = sys.argv[1]
    
    yolo_dataset = Build_Train_Dataset(anno_file,anno_file_type="train", img_size=608)
    dataloader = DataLoader(yolo_dataset, shuffle=True, batch_size=4, num_workers=0)

    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate([label_sbbox.reshape(-1, 26), label_mbbox.reshape(-1, 26),
                                         label_lbbox.reshape(-1, 26)], axis=0)
                labels_mask = labels[..., 4]>0
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                                        axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
