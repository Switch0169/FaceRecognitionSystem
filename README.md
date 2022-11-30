# Face Recognition System
## Introduction
This Project is based on [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) & [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace). Completed the fuction from face detection to face recognition using CenterFace & Inception-ResNet.

## Dataset & Model
Due to device limitations, I did not retrain model or train on the other dataset. If you want to do so, please follow  [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) & [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace).

face_recognition_v1.py uses MTCNN for face detection as same as [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe), but because of slow inference speed, [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace) is used as the detector in face_recognition_v2.py

Inception—Resenet model exceeds GitHub's file size limit, but it could be download from [ZhaoJ9014/face.evoLVe Model Zoo](https://github.com/ZhaoJ9014/face.evoLVe) 
## Performance
Store Face images in the following path
```
├───FaceDatabase
│   ├───Face
│   │   ├───Name1
│   │   ├───Name2
│   │   └───Name3
```

Run face_align.py , and found Aligned Face in the following path

```
│   ├───Face_Aligned
│   │   ├───Name1
│   │   ├───Name2
│   │   └───Name3
```
And the extracted feature will be saved as npy file


Origin Face
<div align="left">
<img src=FaceDatabase\Face\Name2\name2.jpg width=30% height=30%/>
</div>
Aligned Face
<div align="left">
<img src=FaceDatabase\Face_Aligned\Name2\name2.jpg width=30% height=30%/>

Real scene face recognition

<div align="left">
<img src=result1.jpg/>
