import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
import time  # Added for potential delay

def detect_and_recognize_faces(image_path, dataset_path):
    """
    Detect and recognize faces in an image using MTCNN for face detection 
    and DeepFace for face recognition.

    Args:
        image_path (str): Path to the input image.
        dataset_path (str): Path to the dataset folder for face recognition.
    """
    # Read the input image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Initialize MTCNN face detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(img)

    if not faces:
        print("No faces detected.")
        return

    # Prepare the image for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Force model initialization before detection
    DeepFace.build_model('Facenet')  # You can change to 'Facenet', 'ArcFace', etc.

    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        if confidence > 0.5:  # Only consider detections with high confidence
            # Crop the face region
            face_crop = img[y:y + height, x:x + width]

            # Save the cropped face temporarily for recognition
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_crop)

            # Recognize the face using DeepFace
            try:
                # Perform face recognition with DeepFace
                result = DeepFace.find(img_path=temp_face_path, db_path=dataset_path, enforce_detection=False)

                # Check if a match is found
                if len(result) > 0 and not result[0].empty:
                    name = result[0].identity.split(os.sep)[-2]  # Extract name from dataset structure
                else:
                    name = "Unknown"
            except Exception as e:
                print(f"Recognition error: {e}")
                name = "Unknown"

            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Remove the temporary cropped face image
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)

    # Display the annotated image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Configuration
image_path = 'week10/test1.jpg'  # Path to the test image
dataset_path = 'week10/dataset/'  # Path to the dataset folder

# Run the face detection and recognition function
detect_and_recognize_faces(image_path, dataset_path)
