import utils.gpu as gpu
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
        self.__conf_threshold = cfg.VAL.CONF_THRESH
        self.__nms_threshold = cfg.VAL.NMS_THRESH
        self.__device = gpu.select_device()
        self.__classes = cfg.DATASET.CLASSES

        # self.__video_path = video_path
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
        for image_path in self.__file_list:
            frame = cv2.imread(image_path)
            # prev_time = timer()
            bboxes_prd = self.__evalter.get_bbox(frame)
            if bboxes_prd.shape[0] != 0:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                visualize_boxes(image=frame, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
            # curr_time = timer()
            # accum_time +=  curr_time - prev_time
            cv2.imwrite(os.path.join(self.__output_dir,os.path.basename(image_path)),frame)
        print("FPS: {:.04f}".format(len(self.__file_list)/self.__evalter.inference_time))
        
    def get_first_10_imgs(self):
        fh = open(self.__label_path)
        image_paths = []
        for line in fh.readlines():
            line = line.rstrip().split()
            if len(image_paths) < 100:
                if len(line) > 1:
                    image_paths.append(os.path.join("/data",line[0]))
            else:
                break
        print(image_paths)
        self.__file_list =  image_paths
        

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

