import cv2
import datetime
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 실시간, 화면 녹화, 캡쳐, 글자삽입, 시간표시
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
font = ImageFont.truetype("fonts/SCDream6.otf", 20)
is_record = False

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

while True:
    now = datetime.datetime.now()
    nowDatetime = now.strftime("%Y-%m-%d %H:%M:%S")
    nowDatetime_path = now.strftime("%Y-%m-%d %H_%M_%S")

    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=3, minSize=(20, 20)
    )

    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(340, 35), color=(0, 0, 0), thickness=-1)

    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    draw.text(xy=(10, 15), text="얼굴 인식" + nowDatetime, font=font, fill=(255, 255, 255))
    frame = np.array(frame)

    key = cv2.waitKey(30)

    if len(faces):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_4)

    if key == ord("r") and is_record == False:
        is_record = True
        video = cv2.VideoWriter(
            "video" + nowDatetime_path + ".avi",
            fourcc,
            10,
            (frame.shape[1], frame.shape[0]),
        )
    elif key == ord("r") and is_record == True:
        is_record = False
        video.release()
    elif key == ord("c"):
        cv2.imwrite("capture " + nowDatetime_path + ".png", frame)
    elif key == ord("q"):
        break
    if is_record == True:
        video.write(frame)
        cv2.circle(
            img=frame, center=(620, 15), radius=5, color=(0, 0, 255), thickness=-1
        )

    cv2.imshow("test", frame)


capture.release()
cv2.destroyAllWindows()
