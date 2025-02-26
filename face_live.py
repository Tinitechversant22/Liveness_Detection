'''Liveness detection is any technique used to detect a spoof attempt 
by determining whether the source of a biometric sample is a
live human being or a fake representation. 
This is done using algorithms that analyze data collected 
from biometric sensors to determine whether the source is live or reproduced.'''

import os
import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from mtcnn import MTCNN

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize models
facenet = FaceNet()
detector = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def is_live_face(landmarks):
    try:
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[374], landmarks[386]]

        def eye_aspect_ratio(eye):
            return abs(eye[0].y - eye[1].y)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        # Blink detection threshold
        return avg_ear > 0.03

    except IndexError:
        return False

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    is_live = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            is_live = is_live_face(landmarks)

    label = "Real" if is_live else "Fake"
    color = (0, 255, 0) if is_live else (0, 0, 255)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
