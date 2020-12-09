import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pycocotools.coco import COCO
from utils.datasets import Build_VAL_Dataset
from utils.utils import *

def val_collate(batch):
    return tuple(zip(*batch))

class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, cfg):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.cfg = cfg
        self.val_dataset = Build_VAL_Dataset(cfg)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=True, num_workers=8,
        #                     pin_memory=True, drop_last=True, collate_fn=val_collate)
        self.coco = self.convert_to_coco_api()
        
        self.ids = self.coco.getImgIds()

        self.class_ids = sorted(self.coco.getCatIds())
        # self.name = name
        self.max_labels = 50
        self.img_size = cfg.VAL.TEST_IMG_SIZE
        self.min_size = 1


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # load image and preprocess
        img_file = os.path.join(self.cfg.DATA_PATH, "images",
                                '{:07d}.png'.format(id_))
        img = cv2.imread(img_file)
        img, info_img = preprocess(img, self.img_size, jitter=0,
                                   random_placing=0)
        img = np.transpose(img / 255., (2, 0, 1))


        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            labels = label2yolobox(labels, info_img, self.img_size, False)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, id_

    def convert_to_coco_api(self):
        """
        """
        print("in function convert_to_coco_api...")
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in range(len(self.val_dataset)):
            # find better way to get target
            img, targets = self.val_dataset[img_idx]
            image_id = targets["image_id"].item()
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['height'] = img.shape[-2]
            img_dict['width'] = img.shape[-1]
            dataset['images'].append(img_dict)
            bboxes = targets["boxes"]
            
            bboxes = bboxes.tolist()
            labels = targets['labels'].tolist()
            areas = targets['area'].tolist()
            iscrowd = targets['iscrowd'].tolist()
            # if 'masks' in targets:
            #     masks = targets['masks']
            #     # make masks Fortran contiguous for coco_mask
            #     masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            # if 'keypoints' in targets:
            #     keypoints = targets['keypoints']
            #     keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = labels[i]
                categories.add(labels[i])
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id
                # if 'masks' in targets:
                #     ann["segmentation"] = coco_mask.encode(masks[i].numpy())
                # if 'keypoints' in targets:
                #     ann['keypoints'] = keypoints[i]
                #     ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
                dataset['annotations'].append(ann)
                ann_id += 1
        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds