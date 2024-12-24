import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

def detect_and_display_faces(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Initialize MTCNN face detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(img)

    # Loop through detected faces and draw bounding boxes
    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        # Draw a rectangle around each detected face
        if confidence > 0.5:  # Only mark detections with high confidence
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Convert the image from BGR (OpenCV format) to RGB (matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

# Run the function
image_path = 'week10/somany.jpg'  # Path to your image
detect_and_display_faces(image_path)
