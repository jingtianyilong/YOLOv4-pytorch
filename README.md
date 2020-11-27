# YOLOv4-pytorch (attentive YOLOv4 and Mobilenetv3 YOLOv4)
This is a PyTorch re-implementation of YOLOv4 architecture based on the [argusswift/YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) repo. Did some modification on the interface to make custom training easier.

# How to use
## Dataset
The code had been heavily modified, so the COCO and VOC format are not supported currently. So pls convert your dataset from whatever format into this specific yolo format. We use text file which describing the path to the image and its bboxes.
Each line would contain only one image and all the annotation about the bbox comes right after the image path. See example below:
```
images/00001.png x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id 
images/00002.png x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id 
```


## Training
Modify the `.yaml` file in `./experiments/`. Remember to modify the anchor box part. You can generate the anchor box by using:
```
python get_anchor.py experiments/your_file.yaml 9
```
Copy the output to the yaml file you create. To train, you can run:
```
python train.py --config_file experiments/your_file.yaml 
```
Notice that the training would use all the available GPU. So change visible_gpu variable if you have specific needs.

## Check example output
To output detection results of the first 20 images from validation set. You can run:
```
visualize_demo.py --config_file experiments/your_file.yaml
```

# References:
- argusswift/YOLOv4-pytorch: https://github.com/argusswift/YOLOv4-pytorch
- AlexeyAB/darknet: https://github.com/AlexeyAB/darknet
- VCasecnikovs/Yet-Another-YOLOv4-Pytorch: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
