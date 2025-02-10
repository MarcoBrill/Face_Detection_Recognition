import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceDetection:
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the face detection model.
        :param min_detection_confidence: Minimum confidence threshold for detection.
        """
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        :param image: Input image in BGR format.
        :return: List of bounding boxes (x, y, width, height) for detected faces.
        """
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)

        # Extract bounding boxes
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                      int(bboxC.width * w), int(bboxC.height * h)
                faces.append((x, y, width, height))
        return faces


class FaceRecognition:
    def __init__(self, known_faces: List[np.ndarray], known_names: List[str]):
        """
        Initialize the face recognition model.
        :param known_faces: List of known face encodings.
        :param known_names: List of names corresponding to the known face encodings.
        """
        self.known_faces = known_faces
        self.known_names = known_names

    def recognize_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Recognize faces in an image.
        :param image: Input image in BGR format.
        :param faces: List of bounding boxes (x, y, width, height) for detected faces.
        :return: List of recognized names.
        """
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Recognize faces
        recognized_names = []
        for (x, y, width, height) in faces:
            face_image = image_rgb[y:y+height, x:x+width]
            face_encoding = face_recognition.face_encodings(face_image)

            if len(face_encoding) > 0:
                matches = face_recognition.compare_faces(self.known_faces, face_encoding[0])
                name = "Unknown"
                if True in matches:
                    name = self.known_names[matches.index(True)]
                recognized_names.append(name)
        return recognized_names


def main():
    # Load known faces and names
    known_faces = [
        face_recognition.face_encodings(face_recognition.load_image_file("person1.jpg"))[0],
        face_recognition.face_encodings(face_recognition.load_image_file("person2.jpg"))[0],
    ]
    known_names = ["Person 1", "Person 2"]

    # Initialize face detection and recognition
    face_detector = FaceDetection(min_detection_confidence=0.7)
    face_recognizer = FaceRecognition(known_faces, known_names)

    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = face_detector.detect_faces(frame)

        # Recognize faces
        names = face_recognizer.recognize_faces(frame, faces)

        # Draw bounding boxes and names
        for (x, y, width, height), name in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the output
        cv2.imshow("Face Detection and Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
