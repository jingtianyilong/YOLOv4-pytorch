# coding=utf-8
import cv2
import random
import numpy as np
import pdb
import torch.nn.functional as F


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
        # image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image_paded, bboxes
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

        
class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes