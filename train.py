import logging, datetime
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from torch.utils.tensorboard import SummaryWriter
from config import cfg
from config import update_config
from utils import cosine_lr_scheduler
from utils.log import Logger
from utils.utils import init_seed
from apex import amp
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs,0),targets


class Trainer(object):
    def __init__(self, log_dir, resume=False):
        init_seeds(0)
        self.fp_16 = cfg.FP16
        self.device = gpu.select_device()
        self.start_epoch = 0
        self.best_mAP = 0.
        self.accumulate = cfg.TRAIN.ACCUMULATE
        self.log_dir = log_dir
        self.weight_path = "yolov4.weights"
        self.multi_scale_train = cfg.TRAIN.MULTI_SCALE_TRAIN
        if self.multi_scale_train:
            print('Using multi scales training')
        else:
            print('train img size is {}'.format(cfg.TRAIN.TRAIN_IMG_SIZE))
        self.train_dataset = data.Build_Train_Dataset(anno_file=cfg.TRAIN.ANNO_FILE, anno_file_type="train", img_size=cfg.TRAIN.TRAIN_IMG_SIZE)

        self.epochs = cfg.TRAIN.YOLO_EPOCHS if cfg.MODEL.MODEL_TYPE == 'YOLOv4' else cfg.TRAIN.Mobilenet_YOLO_EPOCHS
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.ACCUMULATE,
                                           num_workers=cfg.TRAIN.NUMBER_WORKERS,
                                           shuffle=True, pin_memory=True)
        self.yolov4 = Build_Model(weight_path="yolov4.weights", resume=resume)

        self.yolov4 = self.yolov4.to(self.device)

        self.optimizer = optim.SGD(self.yolov4.parameters(), lr=cfg.TRAIN.LR_INIT,
                                   momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        self.criterion = YoloV4Loss(anchors=cfg.MODEL.ANCHORS, strides=cfg.MODEL.STRIDES,
                                    iou_threshold_loss=cfg.TRAIN.IOU_THRESHOLD_LOSS)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN.LR_INIT,
                                                          lr_min=cfg.TRAIN.LR_END,
                                                          warmup=cfg.TRAIN.WARMUP_EPOCHS*len(self.train_dataloader))
        if resume: self.__load_resume_weights()

    def __load_resume_weights(self):

        last_weight = os.path.join(log_dir,"checkpoints", "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt['model'])

        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(log_dir,"checkpoints", "best.pt")
        last_weight = os.path.join(log_dir,"checkpoints", "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov4.module.state_dict() if torch.cuda.device_count()>1 else self.yolov4.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        # if epoch > 0 and epoch % 10 == 0:
        #     torch.save(chkpt, os.path.join(log_dir,"checkpoints", 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        global writer
        logger.info("Training start,img size is: {:d},batchsize is: {:d}, subdivision: {:d}, worker number is {:d}".format(cfg.TRAIN.TRAIN_IMG_SIZE, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.ACCUMULATE, cfg.TRAIN.NUMBER_WORKERS))
        logger.info(self.yolov4)
        n_train = len(self.train_dataset)
        n_step = n_train // (cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.ACCUMULATE) + 1
        n_remainder = n_train % (cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.ACCUMULATE)
        logger.info("Train datasets number is : {}".format(n_train))

        if self.fp_16: self.yolov4, self.optimizer = amp.initialize(self.yolov4, self.optimizer, opt_level='O1', verbosity=0)

        if torch.cuda.device_count() > 1: self.yolov4 = torch.nn.DataParallel(self.yolov4)
        logger.info("\n===============  start  training   ===============")
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.yolov4.train()
            with tqdm(total=n_train, unit="imgs", desc=f'Epoch {epoch}/{self.epochs}', ncols=30) as pbar:
                for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):

                    imgs = imgs.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    sbboxes = sbboxes.to(self.device)
                    mbboxes = mbboxes.to(self.device)
                    lbboxes = lbboxes.to(self.device)

                    p, p_d = self.yolov4(imgs)

                    loss, loss_ciou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                    label_lbbox, sbboxes, mbboxes, lbboxes)

                    if self.fp_16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    # Accumulate gradient for x batches before optimizing
                    if i % self.accumulate == 0:
                        self.scheduler.step(n_step*epoch + i)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Print batch results
                    if i % (5*self.accumulate) == 0:
                        logger.info("{:3}: total_loss:{:.4f} | loss_ciou:{:.4f} | loss_conf:{:.4f} | loss_cls:{:.4f} | lr:{:.6f}".format(
                            self.train_dataset.img_size, loss, loss_ciou, loss_conf, loss_cls, self.optimizer.param_groups[0]['lr']
                        ))
                        writer.add_scalar('train/loss_ciou', loss_ciou, n_step * epoch + i)
                        writer.add_scalar('train/loss_conf', loss_conf, n_step * epoch + i)
                        writer.add_scalar('train/loss_cls', loss_cls, n_step * epoch + i)
                        writer.add_scalar('train/train_loss', loss, n_step * epoch + i)
                        writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], n_step * epoch + i)
                        # pbar.set_postfix(**{"img_size": self.train_dataset.img_size,
                        #                 "total_loss": float(loss),
                        #                 "loss_ciou": float(loss_ciou),
                        #                 "loss_conf": float(loss_conf),
                        #                 "loss_cls": float(loss_ciou),
                        #                 "lr": float(self.optimizer.param_groups[0]['lr'])})
                    # multi-sclae training (320-608 pixels) every 10 batches
                    if self.multi_scale_train and (i+1) % (5*self.accumulate) == 0:
                        self.train_dataset.img_size = random.choice(range(10, 20)) * 32
                    pbar.update(imgs.shape[0])
                
            mAP = 0.
            evaluator = COCOAPIEvaluator(cfg=cfg,
                                            img_size=cfg.VAL.TEST_IMG_SIZE,
                                            confthre=cfg.VAL.CONF_THRESH,
                                            nmsthre=cfg.VAL.NMS_THRESH)
            coco_stat = evaluator.evaluate(self.yolov4)
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
            writer.add_scalar('val/mAP_50_95',  coco_stat[0], epoch)
            writer.add_scalar('val/mAP_50',     coco_stat[1], epoch)
            writer.add_scalar('val/mAP_75',     coco_stat[2], epoch)
            writer.add_scalar('val/mAP_small',  coco_stat[3], epoch)
            writer.add_scalar('val/mAP_medium', coco_stat[4], epoch)
            writer.add_scalar('val/mAP_large',  coco_stat[5], epoch)
            writer.add_scalar('val/mAR_max_1',  coco_stat[6], epoch)
            writer.add_scalar('val/mAR_max_10', coco_stat[7], epoch)
            writer.add_scalar('val/mAR_max_100',coco_stat[8], epoch)
            writer.add_scalar('val/mAR_small',  coco_stat[9], epoch)
            writer.add_scalar('val/mAR_medium', coco_stat[10], epoch)
            writer.add_scalar('val/mAR_large',  coco_stat[11], epoch)

            self.__save_model_weights(epoch, coco_stat[0])
            logger.info('save weights done')
        
            end = time.time()
            logger.info("cost time:{:.4f}s".format(end - start))
        logger.info("=====Training Finished.   best_test_mAP:{:.3f}%====".format(self.best_mAP))
        
def init_logger(log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    log_file = 'log_' + get_date_str() + '.txt'
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
    parser.add_argument('--resume', type=bool, default=False, help="whether resume training")
    parser.add_argument('--config_file', type=str, default="experiment/demo.yaml", help="yaml configuration file")
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--fp_16', type=bool, default=False, help='whither to use fp16 precision')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    global logger, writer
    args = getArgs()
    # import config file and save it to log
    update_config(cfg, args)
    init_seed(cfg.SEED)
    log_dir = os.path.join("log",os.path.basename(args.config_file)[:-5])

    # if not os.path.exists(log_dir):
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(os.path.join(log_dir,"checkpoints"),exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger = init_logger(log_dir=log_dir)
    logger.debug(cfg)

    Trainer(log_dir,resume= args.resume).train()
