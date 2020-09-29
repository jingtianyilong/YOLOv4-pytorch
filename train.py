import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.cfgim as cfgim
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
    def __init__(self, log_dir, weight_path=yolov4.conv.137.pth, resume=False, accumulate=2, fp_16=True):
        init_seeds(0)
        self.fp_16 = fp_16
        self.device = gpu.select_device()
        self.start_epoch = 0
        self.best_mAP = 0.
        self.accumulate = accumulate
        self.log_dir = log_dir
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN.MULTI_SCALE_TRAIN
        if self.multi_scale_train:
            print('Using multi scales training')
        else:
            print('train img size is {}'.format(cfg.TRAIN.TRAIN_IMG_SIZE))
        self.train_dataset = data.Build_Dataset(anno_file=cfg.TRAIN.ANNO_FILE, anno_file_type="train", img_size=cfg.TRAIN.TRAIN_IMG_SIZE)
        self.epochs = cfg.TRAIN.YOLO_EPOCHS if cfg.MODEL.MODEL_TYPE == 'YOLOv4' else cfg.TRAIN.Mobilenet_YOLO_EPOCHS
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN.BATCH_SIZE,
                                           num_workers=cfg.TRAIN.NUMBER_WORKERS,
                                           shuffle=True, pin_memory=True)

        self.yolov4 = Build_Model(weight_path=weight_path, resume=resume)
        if torch.cuda.device_count()>1:
            self.yolov4 = torch.nn.DataParallel(self.yolov4)
        self.yolov4 = self.yolov4.to(self.device)

        self.cfgimizer = cfgim.SGD(self.yolov4.parameters(), lr=cfg.TRAIN.LR_INIT,
                                   momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        self.criterion = YoloV4Loss(anchors=cfg.MODEL.ANCHORS, strides=cfg.MODEL.STRIDES,
                                    iou_threshold_loss=cfg.TRAINIOU_THRESHOLD_LOSS)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.cfgimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN.LR_INIT,
                                                          lr_min=cfg.TRAIN.LR_END,
                                                          warmup=cfg.TRAIN.WARMUP_EPOCHS*len(self.train_dataloader))
        if resume: self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):

        last_weight = os.path.join(log_dir,"checkpoints", "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt['model'])

        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['cfgimizer'] is not None:
            self.cfgimizer.load_state_dict(chkpt['cfgimizer'])
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
                 'cfgimizer': self.cfgimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(log_dir,"checkpoints", 'backup_epoch%g.pt'%epoch))
        del chkpt



    def train(self):
        global writer
        logger.info("Training start,img size is: {:d},batchsize is: {:d},work number is {:d}".format(cfg.TRAIN.TRAIN_IMG_SIZE,cfg.TRAIN.BATCH_SIZE,cfg.TRAIN.NUMBER_WORKERS))
        logger.info(self.yolov4)
        logger.info("Train datasets number is : {}".format(len(self.train_dataset)))

        if self.fp_16: self.yolov4, self.cfgimizer = amp.initialize(self.yolov4, self.cfgimizer, cfg_level='O1', verbosity=0)
        logger.info("        =======  start  training   ======     ")
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.yolov4.train()

            mloss = torch.zeros(4)
            logger.info("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):
                self.scheduler.step(len(self.train_dataloader)/(cfg.TRAIN.BATCH_SIZE)*epoch + i)

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
                    with amp.scale_loss(loss, self.cfgimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # Accumulate gradient for x batches before cfgimizing
                if i % self.accumulate == 0:
                    self.cfgimizer.step()
                    self.cfgimizer.zero_grad()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:
                    logger.info("  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                        epoch, self.epochs,i, len(self.train_dataloader) - 1, self.train_dataset.img_size,mloss[3], mloss[0], mloss[1],mloss[2],self.cfgimizer.param_groups[0]['lr']
                    ))
                    writer.add_scalar('loss_ciou', mloss[0],
                                      len(self.train_dataloader) / (cfg.TRAIN.BATCH_SIZE) * epoch + i)
                    writer.add_scalar('loss_conf', mloss[1],
                                      len(self.train_dataloader) / (cfg.TRAIN.BATCH_SIZE) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[2],
                                      len(self.train_dataloader) / (cfg.TRAIN.BATCH_SIZE) * epoch + i)
                    writer.add_scalar('train_loss', mloss[3],
                                      len(self.train_dataloader) / (cfg.TRAIN.BATCH_SIZE) * epoch + i)
                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(range(10, 20)) * 32

                mAP = 0.
                if epoch >= 0:
                    logger.info("===== Validate =====".format(epoch, self.epochs))
                    with torch.no_grad():
                        APs, inference_time = Evaluator(self.yolov4, showatt=False).APs_voc()
                        for i in APs:
                            logger.info("{} --> mAP : {}".format(i, APs[i]))
                            mAP += APs[i]
                        mAP = mAP / self.train_dataset.num_classes
                        logger.info("mAP : {}".format(mAP))
                        logger.info("inference time: {:.2f} ms".format(inference_time))
                        writer.add_scalar('mAP', mAP, epoch)
                        self.__save_model_weights(epoch, mAP)
                        logger.info('save weights done')
                    logger.info("  ===test mAP:{:.3f}".format(mAP))
           
            end = time.time()
            logger.info("  ===cost time:{:.4f}s".format(end - start))
        logger.info("=====Training Finished.   best_test_mAP:{:.3f}%====".format(self.best_mAP))
        
def getArgs():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default="experiment/demo.yaml", help="yaml configuration file")
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before cfgimizing')
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
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir= log_dir)
    logger = Logger(log_file_name=os.path.join(log_dir,'log.txt'), log_level=logging.DEBUG, logger_name='YOLOv4').get_log()

    Trainer(log_dir).train()