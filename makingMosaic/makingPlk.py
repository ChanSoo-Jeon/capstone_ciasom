import cv2
import os
from os import listdir
from PIL import Image
from numpy import asarray, expand_dims
import mediapipe as mp
from keras_facenet import FaceNet
import pickle

MyFaceNet = FaceNet()

# 미디어파이프의 FaceDetection 모듈을 초기화합니다.
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

folder = 'photo'
database = {}

for filename in listdir(folder):
    path = os.path.join(folder, filename)
    gbr1 = cv2.imread(path)

    # 미디어파이프를 사용하여 얼굴을 검출합니다.
    rgb_frame = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = gbr1.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
               int(bboxC.width * iw), int(bboxC.height * ih)

        x1, y1, width, height = bbox
    else:
        x1, y1, width, height = 1, 1, 10, 10

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)
    gbr_array = asarray(gbr)

    face = gbr_array[y1:y2, x1:x2]

    face = Image.fromarray(face)
    face = face.resize((160, 160))
    face = asarray(face)

    face = expand_dims(face, axis=0)
    signature = MyFaceNet.embeddings(face)

    database[os.path.splitext(filename)[0]] = signature

    myfile = open("data.pkl", "wb")
    pickle.dump(database, myfile)
    myfile.close()

    print(database)
