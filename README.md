# YOLOv4-pytorch (designed for custom dataset training)
This is a PyTorch re-implementation of YOLOv4 architecture based on the [argusswift/YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) repo. Did some modification on the interface to make custom training easier.

## Branch suggestion
`Master` branch provides reliable train/val/test loop so far. Notice that we have a fix seed in the `class Trainer`. You should comment out this line before using to acquire the seed you set in the yaml file.
```
class Trainer(object):
    def __init__(self, log_dir, resume=False, fine_tune=False):
        init_seeds(0) ## COMMENT OUT THIS LINE
```

Currently, Mosaic lies on a separate branch `mosaic`, which is a little different to the master branch. Also, we eliminate mix up and speed up the data augmentation. We plan to merge this back to master after all the development and testing finished. It might still be buggy now. You can use this beta version by:
```
git checkout mosaic
``` 


# How to use
## Dataset
The code had been heavily modified, so the COCO and VOC format are not supported currently. So pls convert your dataset from whatever format into this specific yolo format. We use text file which describing the path to the image and its bboxes.
Each line would contain only one image and all the annotation about the bbox comes right after the image path. See example below:
```
images/00001.png x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id 
images/00002.png x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id 
```
`x1, y1` refer to the coordinates of top left corner.
`x2, y2` refer to the coordinates of bottom right corner.
All the coordinates are pixel wise location on the original image.

To make sure that the COCOAPI works properly, you might also have to change your image name to a number e.g. `0000234.png`. Or else you might modify the `get_image_id` in `utils/datasets.py` and other function related to index respectively.

## Training
Modify the `.yaml` file in `./experiments/`. Remember to modify the anchor box part. You can generate the anchor box by using:
```
python get_anchor.py experiments/your_file.yaml 9
```
Copy the output to the yaml file you create. To train, you can run:
```
python train.py --config_file experiments/your_file.yaml 
```
Notice that the training would use all available GPU. So change `visible_gpu` variable if you have specific needs.

We also provide a proper way to fine tune your model. You should first train your original model. The script would find your best checkpoint before fine tune and keep on training for several epochs. You should modify the settings in "FINT_TUNE" in your yaml file. `LR_INIT` in fine tune part should be a lot smaller than which before fine tune.
```
FINE_TUNE:
  ANNO_FILE: "fine_tune.txt"
  YOLO_EPOCHS: 50
  LR_INIT: 1e-3
  LR_END: 5e-5
  WARMUP_EPOCHS: 0
```


## Validation
Validation goes automatically every epoch, and it is based on COCO API. Basically, we predict every images in the validation set and generate a COCO format result and submit it to the COCO Toolkit. 

## Test
For testing, simply run:
```
python test.py --config_file experiments//your_file.yaml --test_anno /data/test.txt
```
This works similar to the validation part. All the setting are the same with the validation. We will output all the COCO results as well.

## Check example output
To output detection results of the first 20 images from validation set. You can run:
```
visualize_demo.py --config_file experiments/your_file.yaml
```

# References:
- argusswift/YOLOv4-pytorch: https://github.com/argusswift/YOLOv4-pytorch
- AlexeyAB/darknet: https://github.com/AlexeyAB/darknet
- VCasecnikovs/Yet-Another-YOLOv4-Pytorch: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
