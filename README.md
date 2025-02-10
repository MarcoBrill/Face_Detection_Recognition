# Face Detection and Recognition Workflow

This repository contains a Python script for face detection and face recognition using modern computer vision techniques. The workflow uses MediaPipe for face detection and the `face_recognition` library for face recognition.

1. **Face Detection**:
   - Uses MediaPipe's face detection model to detect faces in real-time.
   - Returns bounding boxes for detected faces.

2. **Face Recognition**:
   - Uses the `face_recognition` library to encode detected faces and compare them with known face encodings.
   - Recognizes faces and assigns names based on the closest match.

## Inputs
Webcam feed or video file.
Pre-loaded images of known faces.

## Outputs
Real-time video feed with bounding boxes around detected faces and their recognized names.

## Requirements
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-detection-recognition.git
   cd face-detection-recognition
