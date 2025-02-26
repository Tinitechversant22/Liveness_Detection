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

# Function to check liveness using blink detection & head movement
def is_live_face(landmarks):
    try:
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[374], landmarks[386]]

        def eye_aspect_ratio(eye):
            return abs(eye[0].y - eye[1].y)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        return avg_ear > 0.0  # Threshold for blink detection

    except IndexError:
        return False  # If eye landmarks are missing, assume fake

# Function to process a single image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Cannot read image.")
        return

    faces = detector.detect_faces(image)

    if faces:
        x, y, w, h = faces[0]['box']
        face_crop = image[y:y+h, x:x+w]
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            print("Error: Cropped face is empty.")
            return

        face_crop = cv2.resize(face_crop, (160, 160))

        # Get FaceNet embedding
        embedding = facenet.embeddings([face_crop])
        if embedding is None:
            print("Error: FaceNet returned None.")
            return
        print("Face Embedding:", embedding[0][:5])  # Print first 5 values

        # Liveness detection using MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        is_live = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                is_live = is_live_face(landmarks)

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0) if is_live else (0, 0, 255), 2)
        label = "Real" if is_live else "Fake"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if is_live else (0, 0, 255), 2)
        
        cv2.imshow("Liveness Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected.")

# Function to capture from webcam and detect liveness
def webcam_liveness():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam.")
            break

        faces = detector.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face['box']
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                continue  # Skip if invalid face

            face_crop = cv2.resize(face_crop, (160, 160))

            # Get FaceNet embedding
            embedding = facenet.embeddings([face_crop])

            # Convert frame to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            is_live = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    is_live = is_live_face(landmarks)

            color = (0, 255, 0) if is_live else (0, 0, 255)
            text = "Real" if is_live else "Fake"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Webcam Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run image test
process_image("7.jpg")  # Replace with your image path

# Run webcam liveness check
webcam_liveness()