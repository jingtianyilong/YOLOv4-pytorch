import shutil
import os
import json
# from tqdm import tqdm


if __name__ == '__main__':
    source_path = "/home/zhirong/Workspace/Code/ecp-preprocess/gt/"
    fake_path = os.path.join(source_path,"real_val")
    if not os.path.exists(fake_path):
        os.makedirs(fake_path)
    split_file = "split.json"
    with open(split_file,"r") as f:
        split_fh = json.load(f)
        fake_images_list = split_fh["test"]
    print("finish read list")
    for image_path in fake_images_list:
        label_path = os.path.join(source_path,image_path.replace("data/images","labels_hrnet_pred_ignore_tag").replace("png","json"))
        dirname, filename = os.path.split(label_path)

        target_dirname = dirname.replace("train","val").replace("labels_hrnet_pred_ignore_tag","real_val")
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)
        shutil.copy2(label_path,os.path.join(target_dirname,filename))