import cv2
import face_recognition
import matplotlib.pyplot as plt
import os

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    
    return known_face_encodings, known_face_names

def detect_and_display_faces(image_path, known_face_encodings, known_face_names):
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    # Loop through detected faces and draw bounding boxes and labels
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around each detected face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw the name of the person
        cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Convert the image from BGR (OpenCV format) to RGB (matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

# Load known faces
known_faces_dir = 'week10/dataset'  # Directory with images of known faces
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Run the function
image_path = 'week10/test1.jpg'  # Path to your image
detect_and_display_faces(image_path, known_face_encodings, known_face_names)
