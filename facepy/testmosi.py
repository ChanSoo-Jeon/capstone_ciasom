import cv2

# 파일, 모자이크
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("video2023-09-21 17_29_42.avi")
v = 20
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        roi = cv2.resize(roi_color, (w // v, h // v))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        frame[y : y + h, x : x + w] = roi

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
