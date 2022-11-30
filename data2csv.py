import torch
import cv2
import numpy as np
import argparse
import os
from img2feature import img2feature
from model_irse import IR_152
from tqdm.auto import tqdm
import pandas as pd


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract feature and save2csv")
    parser.add_argument("-source_root", "--source_root", help="specify your source dir",
                        default="./FaceDatabase/Face_Aligned",
                        type=str)
    parser.add_argument("-dest_root", "--dest_root", help="specify your destination dir",
                        default="./FaceDatabase/Feature",
                        type=str)
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root  # specify your destination dir

    input_size = [112, 112]

    # 读取模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = IR_152(input_size=input_size)
    backbone.load_state_dict(torch.load(
        r"./Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth",
        map_location=device))
    backbone.to(device)
    backbone.eval()

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    df = pd.DataFrame(columns=['feature_path', 'name'])

    names = []
    feature_path = []
    for subfolder in tqdm(os.listdir(source_root)):
        for image_name in os.listdir(os.path.join(source_root, subfolder).replace('\\', '/')):
            img = cv2.imread(os.path.join(source_root, subfolder, image_name).replace('\\', '/'))
            try:  # Handle exception
                feature = img2feature(img, backbone)
                feature = feature.cpu().numpy()
                np.save(dest_root + '/' + subfolder + '.npy', feature)
            except Exception:
                print("{} is discarded due to exception!".format(
                    os.path.join(source_root, subfolder, image_name).replace('\\', '/')))
                continue
            names.append(subfolder)
            feature_path.append(dest_root + '/' + subfolder + '.npy')

    df["feature_path"] = feature_path
    df["name"] = names

    df.to_csv(dest_root + "/Face_Features.csv")
