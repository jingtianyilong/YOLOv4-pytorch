import sys, os
from config import cfg
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


if __name__ == "__main__":
    config_file = sys.argv[1]
    num_of_anchor = int(sys.argv[2])
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    print(cfg)
    f = open(os.path.join(cfg.DATA_PATH,cfg.TRAIN.ANNO_FILE))       
    lines = [line.rstrip('\n') for line in f.readlines()]
    annotation_dims = []
    for line in lines:
        line = line.rstrip().split()
        img = Image.open(os.path.join(cfg.DATA_PATH,line[0]))
        img_w,img_h = img.size
        try:
            for obj in line[1:]:
                obj = obj.split(",")
                bbox_w = (float(obj[2]) - float(obj[0])) / img_w * cfg.TRAIN.TRAIN_IMG_SIZE
                bbox_h = (float(obj[3]) - float(obj[1])) / img_h * cfg.TRAIN.TRAIN_IMG_SIZE
                annotation_dims.append([bbox_w, bbox_h])
        except:
            pass
    annotation_dims = np.array(annotation_dims)
    kmeans_calc = KMeans(n_clusters=num_of_anchor)
    kmeans_calc.fit(annotation_dims)
    y_kmeans = kmeans_calc.predict(annotation_dims)
    anchor_list = []
    for ind in range(num_of_anchor):
        anchor_list.append(np.mean(annotation_dims[y_kmeans==ind],axis=0).astype(np.int16))
    anchor_list=np.array(anchor_list)
    anchor_list.sort(axis=0)
    # anchor_list = np.array(list(map(int,np.concatenate(anchor_list))))
    small = anchor_list[:int(num_of_anchor/3)] / 8
    medium = anchor_list[int(num_of_anchor/3):-int(num_of_anchor/3)] / 16
    large = anchor_list[-int(num_of_anchor/3):] / 32
    
    print([small.tolist(),medium.tolist(), large.tolist()])

    
