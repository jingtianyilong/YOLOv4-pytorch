import os
from yacs.config import CfgNode as CN

_C = CN()
_C.SEED = 358
_C.DATA_PATH = "/data"
_C.PROJECT_PATH = "/data"
_C.DETECTION_PATH = "/"
_C.FP16 = False

# model
_C.MODEL = CN()
_C.MODEL.ANCHORS= [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
                   [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
                   [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]]  # Anchors for big obj(142,110),(192,243),(459,401)
_C.MODEL.STRIDES = [8, 16, 32]
_C.MODEL.ANCHORS_PER_SCLAE = 3
_C.MODEL.MODEL_TYPE = 'YOLOv4'  #YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4
_C.MODEL.CONV_TYPE = 'DO_CONV'  #conv type:DO_CONV or GENERAL
_C.MODEL.ATTENTION = 'NONE'  #attention type:SEnet、CBAM or NONE

# train
_C.TRAIN = CN()
_C.TRAIN.ANNO_FILE = "train.txt"
_C.TRAIN.DATA_TYPE = 'Customer'  #DATA_TYPE: VOC ,COCO or Customer
_C.TRAIN.TRAIN_IMG_SIZE = 608
_C.TRAIN.AUGMENT = True
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.MULTI_SCALE_TRAIN = True
_C.TRAIN.IOU_THRESHOLD_LOSS = 0.5
_C.TRAIN.YOLO_EPOCHS = 80
_C.TRAIN.Mobilenet_YOLO_EPOCHS = 120
_C.TRAIN.NUMBER_WORKERS = 16
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0005
_C.TRAIN.LR_INIT = 1e-4
_C.TRAIN.LR_END = 1e-6
_C.TRAIN.WARMUP_EPOCHS = 2  # or None



# val
_C.VAL =  CN()
_C.VAL.ANNO_FILE = "val.txt"
_C.VAL.TEST_IMG_SIZE = 608
_C.VAL.BATCH_SIZE =  1
_C.VAL.NUMBER_WORKERS =  8
_C.VAL.CONF_THRESH =  0.5
_C.VAL.NMS_THRESH =  0.45
_C.VAL.MULTI_SCALE_VAL =  False
_C.VAL.FLIP_VAL =  False
_C.VAL.Visual =  True


_C.DATASET = CN()
_C.DATASET.NUM =  1 #your dataset number
_C.DATASET.CLASSES = ['aeroplane'] # your dataset class


_C.VOC_DATA = CN()
_C.VOC_DATA.NUM = 20
_C.VOC_DATA.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']


_C.COCO_DATA = CN()
_C.COCO_DATA.NUM = 80
_C.COCO_DATA.CLASSES = ['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'dining table',
'toilet',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush']

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
