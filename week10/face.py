import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust parameters and detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Adjust scale factor slightly
        minNeighbors=3,    # Lower minNeighbors for sensitivity
        minSize=(20, 20)   # Adjust minSize for smaller faces
    )

    # Check if any faces are found
    if len(faces) > 0:
        print("Face(s) detected in the image.")
    else:
        print("No face detected in the image.")

# Run the function
image_path = 'week10/3cat.jpg'  # Path to your image
detect_face(image_path)
