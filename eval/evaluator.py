import shutil
from eval import voc_eval
from utils.data_augment import *
from utils.tools import *
from tqdm import tqdm
from utils.visualize import *
from utils.heatmap import imshowAtt
from config import cfg
import time
current_milli_time = lambda: int(round(time.time() * 1000))
class Evaluator(object):
    def __init__(self, model, showatt):
        if cfg.TRAIN.DATA_TYPE == 'VOC':
            self.classes = cfg.VOC_DATA.CLASSES
        elif cfg.TRAIN.DATA_TYPE == 'COCO':
            self.classes = cfg.COCO_DATA.CLASSES
        else:
            self.classes = cfg.DATASET.CLASSES
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'pred_result')
        self.val_data_path = os.path.join(cfg.DATA_PATH, 'VOCtest-2007', 'VOCdevkit', 'VOC2007')
        self.conf_thresh = cfg.VAL.CONF_THRESH
        self.nms_thresh = cfg.VAL.NMS_THRESH
        self.val_shape = cfg.VAL.TEST_IMG_SIZE
        self.model = model
        self.device = next(model.parameters()).device
        self.__visual_imgs = 0
        self.showatt = showatt
        self.inference_time = 0.

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)

        txtpath = "./output/detection-results/"
        if not os.path.exists(txtpath):
            os.mkdir(txtpath)
        os.mkdir(self.pred_result_path)
        print('val img size is {}'.format(self.val_shape))
        for img_ind in tqdm(img_inds):
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind+'.jpg')
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            f = open("./output/detection-results/" + img_ind + ".txt", "w")
            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as r:
                    r.write(s)
                f.write("%s %s %s %s %s %s\n" % (class_name, score, str(xmin), str(ymin), str(xmax), str(ymax)))
            f.close()
        self.inference_time = 1.0 * self.inference_time / len(img_inds)
        return self.__calc_APs(), self.inference_time

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            if self.showatt: _, p_d, beta = self.model(img)
            else: _, p_d = self.model(img)
            self.inference_time += (current_milli_time() - start_time)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        if self.showatt and len(img):
            self.__show_heatmap(beta[2], org_img)
        return bboxes

    def __show_heatmap(self, beta, img):
        imshowAtt(beta, img)

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        Filter out the prediction box to remove the unreasonable scale of the box
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)Crop off the portion of the predicted Bbox that is beyond the original image
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)Sets the coor of an invalid bbox to 0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)Remove bboxes that are not in the valid range
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)Remove bboxes whose score is below the score_threshold
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        Calculate ap values for each category
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        # annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        annopath = os.path.join(self.val_data_path, 'Annotations\\' + '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path,  'ImageSets', 'Main', 'test.txt')
        APs = {}
        Recalls = {}
        Precisions = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            Recalls[cls] = R
            Precisions[cls] = P
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
