import logging, datetime
import utils.gpu as gpu
from model.build_model import Build_Model
from tqdm import tqdm
import torch
import time
import argparse
from eval.evaluator import *
from utils.tools import *
from config import cfg
from config import update_config
from utils.log import Logger
from utils.utils import init_seed
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator

class Tester(object):
    def __init__(self, log_dir):
        init_seeds(0)
        self.device = gpu.select_device()
        self.log_dir = log_dir
        self.yolov4 = Build_Model(weight_path=None, resume=False)
        self.yolov4 = self.yolov4.to(self.device)
        self.__load_best_weights()

    def __load_best_weights(self):
        best_weight = os.path.join(log_dir,"checkpoints", "best.pt")
        chkpt = torch.load(best_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt)
        del chkpt

    def test(self):
        logger.info(self.yolov4)
        evaluator = COCOAPIEvaluator(cfg=cfg,
                                img_size=cfg.VAL.TEST_IMG_SIZE,
                                confthre=cfg.VAL.CONF_THRESH,
                                nmsthre=cfg.VAL.NMS_THRESH)
        logger.info("\n===============  start  testing   ===============")
        start = time.time()
        coco_stat = evaluator.evaluate(self.yolov4)
        end = time.time()

        logger.info("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.04f}".format(coco_stat[0]))
        logger.info("Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.04f}".format(coco_stat[1]))            
        logger.info("Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.04f}".format(coco_stat[2]))            
        logger.info("Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.04f}".format(coco_stat[3]))            
        logger.info("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.04f}".format(coco_stat[4]))            
        logger.info("Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.04f}".format(coco_stat[5]))            
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.04f}".format(coco_stat[6]))            
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.04f}".format(coco_stat[7]))            
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.04f}".format(coco_stat[8]))            
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.04f}".format(coco_stat[9]))            
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.04f}".format(coco_stat[10])) 
        logger.info("Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.04f}".format(coco_stat[11]))             
        logger.info("cost time:{:.4f}s".format(end - start))
        logger.info("FPS:{:.4f}".format(len(evaluator.dataloader)/(end - start)))
        
def init_logger(log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    log_file = 'test_log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging    

def getArgs():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default="experiment/demo.yaml", help="yaml configuration file")
    parser.add_argument("--test_anno",type=str, default=None, help="test annotation file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = getArgs()
    # import config file and save it to log
    update_config(cfg, args)
    cfg.defrost()
    cfg.VAL.ANNO_FILE = args.test_anno
    cfg.freeze()
    init_seed(cfg.SEED)
    log_dir = os.path.join("log",os.path.basename(args.config_file)[:-5])

    logger = init_logger(log_dir=log_dir)
    logger.debug(cfg)

    Tester(log_dir).test()
