from face_detection_recognition import FaceDetection, FaceRecognition
import cv2

# Load known faces and names
known_faces = [
    face_recognition.face_encodings(face_recognition.load_image_file("person1.jpg"))[0],
    face_recognition.face_encodings(face_recognition.load_image_file("person2.jpg"))[0],
]
known_names = ["Person 1", "Person 2"]

# Initialize face detection and recognition
face_detector = FaceDetection(min_detection_confidence=0.7)
face_recognizer = FaceRecognition(known_faces, known_names)

# Load an image
image = cv2.imread("test_image.jpg")

# Detect faces
faces = face_detector.detect_faces(image)

# Recognize faces
names = face_recognizer.recognize_faces(image, faces)

# Draw bounding boxes and names
for (x, y, width, height), name in zip(faces, names):
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save or display the result
cv2.imwrite("output_image.jpg", image)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
