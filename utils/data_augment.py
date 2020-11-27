# coding=utf-8
import cv2
import random
import numpy as np
import pdb
import torch.nn.functional as F
from tools import *

def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                    left_shift, right_shift, top_shift, bot_shift):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]
    return out_img, bboxes

def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    if bboxes != []:
        bboxes[:, 0] -= dx
        bboxes[:, 2] -= dx
        bboxes[:, 1] -= dy
        bboxes[:, 3] -= dy

        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

        out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                                ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                                ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                                ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
        list_box = list(range(bboxes.shape[0]))
        for i in out_box:
            list_box.remove(i)
        bboxes = bboxes[list_box]

        bboxes[:, 0] += xd
        bboxes[:, 2] += xd
        bboxes[:, 1] += yd
        bboxes[:, 3] += yd
        return bboxes
    else:
        return bboxes

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            img = img[:, ::-1, :]
            
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes

class RandomHSVAug(object):
    def __init__(self, hue_jitter, bright_jitter, sat_jitter, p=0.5):
        self.p = p
        self.hue_jitter = hue_jitter
        self.bright_jitter = bright_jitter
        self.sat_jitter = sat_jitter
        
    def __call__(self,img):
        if random.random() < self.p:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
            img[:,:,0] *= random.uniform(1-self.hue_jitter, 1+self.hue_jitter)
            img[:,:,1] *= random.uniform(1-self.sat_jitter, 1+self.sat_jitter)
            img[:,:,2] *= random.uniform(1-self.bright_jitter, 1+self.bright_jitter)
            img = cv2.cvtColor(np.uint8(img),cv2.COLOR_HSV2BGR)
        return img

class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, correct_box=True):
        self.correct_box = correct_box

    def __call__(self, img, bboxes, target_shape):
        h_org , w_org , _= img.shape
        h_target, w_target = target_shape
        img = img.astype(np.float32)

        resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((h_target, w_target, 3), 128.0)
        dw = int((w_target - resize_w) / 2)
        dh = int((h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if len(bboxes_org) > 0:
            if random.random() > self.p:
                lam = np.random.beta(1.5, 1.5)
                # print("img_org",img_org.shape,"img_mix",img_mix.shape)
                img = lam * img_org + (1 - lam) * img_mix
                
                bboxes_org = np.concatenate(
                    [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
                bboxes_mix = np.concatenate(
                    [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
                bboxes = np.concatenate([bboxes_org, bboxes_mix])

            else:
                img = img_org
                bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        else:
            img = img_org
            bboxes = bboxes_org
            
        return img, bboxes

class Mosaic(object):
    """
    reference: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch/blob/c8cc941e08b16212d0be715d92d68ccc061da554/dataset.py
    Thank VCasecnikovs for his great work on mosaic augmentation. It's a 
    more official implementation since AlexeyAB also on the contributer 
    list.
    """
    def __init__(self, cross_offset=0.2):
        self.cross_offset = cross_offset     
    def __call__(self, img_org, bboxes_org,m_img_1, m_bboxex_1,m_img_2, m_bboxes_2,m_img_3, m_bboxes_3):
        mossaic_img = torch.zeros(3, self.img_size, self.img_size)
        
        # FINDING CROSS POINT
        cross_x = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))
        cross_y = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))

        fragment_img, fragment_bbox = self.get_mosaic(0, cross_x, cross_y, tensor_img, boxes)
        mossaic_img[:, 0:cross_y, 0:cross_x] = fragment_img
        boxes = fragment_bbox
        return img, bboxes
    def get_mosaic(self, n, cross_x, cross_y, tensor_img, boxes):


        t_height = tensor_img.shape[1]
        t_width = tensor_img.shape[2]

        xyxy_bboxes = xywh2xyxy(boxes[:, 1:])

        relative_cross_x = cross_x / self.img_size
        relative_cross_y = cross_y / self.img_size

        #CALCULATING TARGET WIDTH AND HEIGHT OF PICTURE
        if n == 0:
            width_of_nth_pic = cross_x 
            height_of_nth_pic = cross_y
        elif n == 1:
            width_of_nth_pic = self.img_size - cross_x
            height_of_nth_pic = cross_y
        elif n == 2:
            width_of_nth_pic = cross_x
            height_of_nth_pic = self.img_size - cross_y
        elif n == 3:
            width_of_nth_pic = self.img_size - cross_x
            height_of_nth_pic = self.img_size - cross_y

        # self.img_size - width_of_1st_pic
        # selg.img_size - height_of_1st_pic  


        # CHOOSING TOP LEFT CORNER (doing offset to have more than fex pixels in bbox :-) )
        cut_x1 = random.randint(0, int(t_width * 0.33))
        cut_y1 = random.randint(0, int(t_height * 0.33))


        # Now we should find which axis should we randomly enlarge (this we do by finding out which ratio is bigger); cross x is basically width of the top left picture
        if (t_width - cut_x1) / width_of_nth_pic < (t_height - cut_y1) / height_of_nth_pic:
            cut_x2 = random.randint(cut_x1 + int(t_width * 0.67), t_width)
            cut_y2 = int(cut_y1 + (cut_x2-cut_x1)/width_of_nth_pic*height_of_nth_pic)

        else:
            cut_y2 = random.randint(cut_y1 + int(t_height * 0.67), t_height)
            cut_x2 = int(cut_x1 + (cut_y2-cut_y1)/height_of_nth_pic*width_of_nth_pic)
        
        # RESIZING AND INSERTING (TO DO 2D interpolation wants 4 dimensions, so I add and remove one by using None and squeeze)
        tensor_img = F.interpolate(tensor_img[:, cut_y1:cut_y2,  cut_x1:cut_x2][None], (height_of_nth_pic, width_of_nth_pic)).squeeze()
        
        # BBOX
        relative_cut_x1 = cut_x1 / t_width
        relative_cut_y1 = cut_y1 / t_height
        relative_cropped_width = (cut_x2 - cut_x1) / t_width
        relative_cropped_height = (cut_y2 - cut_y1) / t_height

        # SHIFTING TO CUTTED IMG SO X1 Y1 WILL 0
        xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] - relative_cut_x1
        xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] - relative_cut_y1
        xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] - relative_cut_x1
        xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] - relative_cut_y1

        # RESIZING TO CUTTED IMG SO X2 WILL BE 1
        xyxy_bboxes[:, 0] /= relative_cropped_width
        xyxy_bboxes[:, 1] /= relative_cropped_height
        xyxy_bboxes[:, 2] /= relative_cropped_width
        xyxy_bboxes[:, 3] /= relative_cropped_height

        # CLAMPING BOUNDING BOXES, SO THEY DO NOT OVERCOME OUTSIDE THE IMAGE
        xyxy_bboxes[:, 0].clamp_(0, 1)
        xyxy_bboxes[:, 1].clamp_(0, 1)
        xyxy_bboxes[:, 2].clamp_(0, 1)
        xyxy_bboxes[:, 3].clamp_(0, 1)

        # FILTER TO THROUGH OUT ALL SMALL BBOXES
        filter_minbbox = (xyxy_bboxes[:, 2] - xyxy_bboxes[:, 0] > self.bbox_minsize) & (xyxy_bboxes[:, 3] - xyxy_bboxes[:, 1] > self.bbox_minsize)

        # RESIZING TO MOSAIC
        if n == 0:
            xyxy_bboxes[:, 0] *= relative_cross_x #
            xyxy_bboxes[:, 1] *= relative_cross_y #(1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= relative_cross_x #
            xyxy_bboxes[:, 3] *= relative_cross_y #(1 - relative_cross_y)
        elif n==1:
            xyxy_bboxes[:, 0] *= (1 - relative_cross_x) 
            xyxy_bboxes[:, 1] *= relative_cross_y
            xyxy_bboxes[:, 2] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 3] *= relative_cross_y
        elif n==2:
            xyxy_bboxes[:, 0] *= relative_cross_x
            xyxy_bboxes[:, 1] *= (1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= relative_cross_x
            xyxy_bboxes[:, 3] *= (1 - relative_cross_y)
        elif n==3:
            xyxy_bboxes[:, 0] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 1] *= (1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 3] *= (1 - relative_cross_y)

        # RESIZING TO MOSAIC
        if n == 0:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] # + relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] # + relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] # + relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] # + relative_cross_y
        elif n==1:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] + relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1]
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] + relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3]
        elif n==2:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0]
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] + relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2]
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] + relative_cross_y
        elif n==3:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] + relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] + relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] + relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] + relative_cross_y
        
        boxes = boxes[filter_minbbox]
        boxes[:, 1:] = xyxy2xywh(xyxy_bboxes)[filter_minbbox]

        return tensor_img, boxes
        
        
class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes