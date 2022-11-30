import torch
import cv2
import numpy as np
from img2feature import img2feature
from model_irse import IR_152
import pandas as pd


def cosine_metric(x1, x2):
    x1, x2 = x1.reshape(-1), x2.reshape(-1)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = IR_152(input_size=(112, 112))
    backbone.load_state_dict(torch.load(
        r"./Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth",
        map_location=device))
    backbone.to(device)
    backbone.eval()

    df = pd.read_csv('FaceDatabase/Feature/Face_Features.csv')
    feature = img2feature(cv2.imread("FaceDatabase/Face_Aligned/jin_zhang/j_z_0.jpg"), backbone)

    max_idx = 0
    max_value = -1
    idx = 0

    base_features = [np.load(path) for path in df["feature_root"].values]
    for f in base_features:
        metric = cosine_metric(feature, f)
        print(metric)
        if cosine_metric(feature, f) > max_value:
            max_idx = idx
            max_value = metric
        idx += 1

    name = df.loc[max_idx, "name"] + " %.4f" % max_value
