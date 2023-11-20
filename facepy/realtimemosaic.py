import cv2

# 실시간, 모자이크, 화면 녹화, 캡쳐
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

is_record = False
v = 20

while True:
    fileox, frame = capture.read()
    if not fileox:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=3, minSize=(20, 20)
    )

    key = cv2.waitKey(30)

    if len(faces):
        for x, y, w, h in faces:
            roi_color = frame[y : y + h, x : x + w]

            roi = cv2.resize(roi_color, (w // v, h // v))
            roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
            frame[y : y + h, x : x + w] = roi

    if key == ord("r") and is_record == False:
        is_record = True
        video = cv2.VideoWriter(
            "video.avi",
            fourcc,
            10,
            (frame.shape[1], frame.shape[0]),
        )

    elif key == ord("r") and is_record == True:
        is_record = False
        video.release()
    elif key == ord("c"):
        cv2.imwrite("capture.png", frame)
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
