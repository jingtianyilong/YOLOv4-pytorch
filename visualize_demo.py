import utils.gpu as gpu
import random
from tqdm import tqdm
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
from timeit import default_timer as timer
from config import cfg, update_config
from utils.visualize import *
from utils.torch_utils import *


class Detection(object):
    def __init__(self,
                 label_path,
                 weight_path=None,
                 output_dir=None,
                 ):
        self.__label_path = os.path.join("/data",label_path)
        self.get_first_10_imgs()
        self.__num_class = cfg.DATASET.NUM
        
        # these should be set still
        self.__conf_threshold = 0.25
        self.__nms_threshold = 0.5
        #######################################
        
        self.__device = gpu.select_device()
        self.__classes = cfg.DATASET.CLASSES
        self.__output_dir = output_dir
        self.__model = Build_Model().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, showatt=False)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def detect_demo(self):
        import cv2

        accum_time = 0
        for image_path in tqdm(self.__file_dict.keys()):
            # print(self.__file_dict[image_path])
            frame = cv2.imread(image_path)
            # prev_time = timer()
            bboxes_prd = self.__evalter.get_bbox(frame)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                visualize_boxes(image=frame, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                visualize_boxes_and_labels_on_image_array(image=frame, boxes=self.__file_dict[image_path][:,:4], classes=self.__file_dict[image_path][:,4], scores=None, line_thickness=10, category_index=self.__classes)
            cv2.imwrite(os.path.join(self.__output_dir,os.path.basename(image_path)),frame)
        print("FPS: {:.04f}".format(1000*len(self.__file_dict.keys())/self.__evalter.inference_time))
        
    def get_first_10_imgs(self):
        fh = open(self.__label_path)
        image_paths = {}
        ####################################
        # for random select
        ####################################
        # random.seed(1)
        # lines = random.choices(fh.readlines(),k=10)

        # for line in lines:
        #     line = line.rstrip().split()
        #     if len(line)>1:
        #         image_paths[os.path.join("/data",line[0])] = np.array([list(map(int,i.split(","))) for i in line[1:]])
        #     else:
        #         break
        ####################################
        # for on demand plot
        ####################################
        lines = fh.readlines()
        imgs = ["images/0021023.png",
                "images/0020485.png",
                "images/0021042.png",
                "images/0021630.png",
                "images/0021729.png",
                "images/0021781.png"]
        for line in lines:
            line = line.rstrip().split()
            if line[0] in imgs:
                image_paths[os.path.join("/data",line[0])] = np.array([list(map(int,i.split(","))) for i in line[1:]])
        #####################################
        self.__file_dict = image_paths
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='experiments/demo.yaml', help='video file path')
    args = parser.parse_args()
    update_config(cfg, args)
    
    log_dir = os.path.join("log",os.path.basename(args.config_file)[:-5])
    output_dir = os.path.join(log_dir, "demo")
    os.makedirs(output_dir, exist_ok=True)
    weight_path = os.path.join(log_dir,"checkpoints","best.pt")
    
    Detection(label_path=cfg.VAL.ANNO_FILE,
              weight_path=weight_path,
              output_dir=output_dir).detect_demo()

