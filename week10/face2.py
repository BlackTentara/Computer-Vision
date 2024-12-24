import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_save_face(image_path, output_path="output_with_faces.jpg"):
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
        print(f"Detected {len(faces)} face(s) in the image.")
        # Loop over the detected faces and draw circles around them
        for (x, y, w, h) in faces:
            # Calculate the center and radius of the circle
            center = (x + w // 2, y + h // 2)
            radius = w // 2  # Use half of the width as the radius
            # Draw the circle on the original image
            cv2.circle(img, center, radius, (0, 255, 0), 2)
    else:
        print("No face detected in the image.")

    # Save the output image with circles around detected faces
    cv2.imwrite(output_path, img)
    print(f"Output image saved as '{output_path}'")

# Run the function
image_path = 'week10/gerry.jpg'  # Path to your image
output_path = 'week10/output_with_faces.jpg'  # Specify where to save the output image
detect_and_save_face(image_path, output_path)
