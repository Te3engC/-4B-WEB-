import cv2
import os
import pickle
import numpy as np
from picamera2 import Picamera2, Preview

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError("error")

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

feature_file = "face_features.pkl"

if os.path.exists(feature_file):
    with open(feature_file, "rb") as f:
        known_features, known_names = pickle.load(f)
else:
    known_features, known_names = [], []

def register_face(name, num_photos=5):
    path = f"known_faces/{name}"
    os.makedirs(path, exist_ok=True)
    count = 0
    while count < num_photos:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{count}.jpg", face_img)
            kp, des = orb.detectAndCompute(face_img, None)
            if des is not None:
                known_features.append(des)
                known_names.append(name)
                count += 1
                print(f"Saved face {count} for {name}")
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    with open(feature_file, "wb") as f:
        pickle.dump((known_features, known_names), f)
    print(f"{name} registration complete and features saved.")
#设置你的面部信息
#register_face("tesengc", num_photos=10)

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            kp, des = orb.detectAndCompute(face_img, None)
            label = "Unknown"
            if des is not None:
                best_score = 0
                for i, known_des in enumerate(known_features):
                    matches = bf.match(des, known_des)
                    score = len(matches)
                    if score > best_score:
                        best_score = score
                        label = known_names[i]
                if best_score < 10:
                    label = "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
