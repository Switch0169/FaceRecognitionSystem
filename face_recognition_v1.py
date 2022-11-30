from PIL import Image, ImageDraw
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import cv2
import numpy as np
import torch
from model_irse import IR_152
import pandas as pd
from img2feature import img2feature


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def cal_fps(t1, t2, f):
    fps = 1 / ((t2 - t1) / f)
    return fps


def show_results(img, bounding_boxes, facial_landmarks=[], names=[]):
    """Draw bounding boxes and facial landmarks with inference names.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
        names: a string numpy array
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    idx = 0
    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='green', width=3)
        draw.text((b[0], b[1] - 10), names[idx], fill=(0, 255, 255))
        idx += 1

    inx = 0
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 2.0, p[i + 5] - 2.0),
                (p[i] + 2.0, p[i + 5] + 2.0)
            ], outline='red')

    return img_copy


def PIL2cv(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def cv2PIL(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_img_aligned(img, landmarks, crop_size, scale):
    """
    Args:
        img: PIL.Image
        landmarks: results of detect_face
        crop_size: int
        scale:  float

    Returns:
        return list of PIL.Image
    """
    images = []
    reference = get_reference_facial_points(default_square=True) * scale
    for i in range(len(landmarks)):
        facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)
        images.append(img_warped)
    return images


def cosine_metric(x1, x2):
    x1, x2 = x1.reshape(-1), x2.reshape(-1)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_most_similar_names(in_features, df):
    threshold = 0.6
    names = []
    db_features = [np.load(path) for path in df["feature_root"].values]
    for i in range(len(in_features)):
        max_idx = 0
        max_prob = 0
        for idx, f in enumerate(db_features):
            metric = cosine_metric(in_features[i], f)
            if metric > max_prob:
                max_idx = idx
                max_prob = metric
        if max_prob >= threshold:
            name = df.loc[max_idx, "name"] + " %.4f" % max_prob
        else:
            name = "unknown"
        names.append(name)
    return names


if __name__ == '__main__':
    crop_size = 112
    scale = crop_size / 112
    input_size = [crop_size, crop_size]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = IR_152(input_size=input_size)
    backbone.load_state_dict(torch.load(
        r'./Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth',
        map_location=device))
    backbone.to(device)
    backbone.eval()

    df = pd.read_csv("FaceDatabase/Feature/Face_Features.csv")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            t1 = cv2.getTickCount()
            frame = cv2PIL(frame)
            try:  # Handle exception
                bounding_boxes, landmarks = detect_faces(frame)
            except Exception:
                frame = PIL2cv(frame)
                cv2.imshow("video", frame)
                continue
            if len(landmarks) != 0 and len(bounding_boxes) != 0:
                images_aligned = get_img_aligned(frame, landmarks, crop_size, scale)
                idx = 0
                features = []
                for img in images_aligned:
                    img = PIL2cv(img)
                    with torch.no_grad():
                        print(img.shape)
                        feature = img2feature(img, backbone)
                        features.append(feature)
                names = get_most_similar_names(features, df)
                frame = show_results(frame, bounding_boxes, landmarks, names=names)

            frame = PIL2cv(frame)
            fps = cal_fps(t1, cv2.getTickCount(), cv2.getTickFrequency())
            cv2.putText(frame, "FPS: %d" % fps, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cv2.destroyAllWindows()
            break
