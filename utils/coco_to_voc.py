from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

# the path you want to save your results for coco to voc
savepath = "cocodata/xml/"
img_dir = savepath + 'images/'
anno_dir = savepath + 'Annotations/'
datasets_list=['train2017', 'val2017', 'test2017']

classes_names = ['person',
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
'toothbrush',]
# Store annotations and train2014/val2014/... in this folder
dataDir = 'E:\code\object_detection\yolov3-other-master\YOLOV3-master\cocodata/'

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


mkr(img_dir)
mkr(anno_dir)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, dataset, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = anno_dir + filename[:-3] + 'xml'
    img_path = dataDir + dataset + '/' + filename
    # print(img_path)
    dst_imgpath = img_dir + filename

    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        # print(filename + " not a RGB image")
        return
    # shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                if xmin <= 0:
                    xmin += 1
                if ymin <= 0:
                    ymin += 1
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs

if __name__ == "__main__":
    i = 0
    for dataset in datasets_list:
        a = datasets_list

        # ./COCO/annotations/instances_train2014.json
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)

        # COCO API for initializing annotated data
        coco = COCO(annFile)
        '''
        COCO finished:
        loading annotations into memory...
        Done (t=0.81s)
        creating index...
        index created!
        end
        '''
        # show all classes in coco
        classes = id2name(coco)
        # print(classes)
        # [1, 2, 3, 4, 6, 8]
        classes_ids = coco.getCatIds(catNms=classes_names)
        # print(classes_ids)
        for cls in classes_names:
            i += 1
            print(i)
            b = classes_names
            # Get ID number of this class
            cls_id = coco.getCatIds(catNms=[cls])
            img_ids = coco.getImgIds(catIds=cls_id)
            # print(cls, len(img_ids))
            # imgIds=img_ids[0:10]
            for imgId in tqdm(img_ids):
                c = img_ids
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
                # print(objs)
                save_annotations_and_imgs(coco, dataset, filename, objs)
