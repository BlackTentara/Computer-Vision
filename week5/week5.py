import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to rotate the image by a given angle
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Function to detect and label Harris corners
def detect_and_label_corners(image, block_size=2, ksize=3, k=0.04, threshold_ratio=0.1, min_distance=10):
    # Convert the image to float32 (required by the cornerHarris function)
    image_float = np.float32(image)

    # Apply Harris Corner Detection
    harris_response = cv2.cornerHarris(image_float, block_size, ksize, k)

    # Dilate the corners to enhance the detected points
    harris_response = cv2.dilate(harris_response, None)

    # Threshold to mark the detected corners on the original image
    threshold = threshold_ratio * harris_response.max()

    # Convert the original image to BGR to visualize the corners in color
    corners_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Mark detected corners in red (255, 0, 0) on the color image
    corners_image_color[harris_response > threshold] = [255, 0, 0]

    # Find corner coordinates dynamically
    corner_coordinates = np.argwhere(harris_response > threshold)

    # Filter the corners to avoid redundancy
    filtered_corners = []
    for corner in corner_coordinates:
        if all(np.linalg.norm(corner - np.array(c)) > min_distance for c in filtered_corners):
            filtered_corners.append(corner)

    # Convert to numpy array for easy indexing
    filtered_corners = np.array(filtered_corners)

    # Generate labels based on the number of corners detected
    labels = [chr(65 + i) for i in range(len(filtered_corners))]

    # Place the labels at the detected corners
    for i, (y, x) in enumerate(filtered_corners):
        cv2.putText(corners_image_color, labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return corners_image_color

# Load the image (grayscale)
image = cv2.imread('week5/square3.jpeg', cv2.IMREAD_GRAYSCALE)

# Define the angle of rotation
angle = 45  # Modify this value to rotate by a different angle

# Rotate the image by the specified angle
rotated_image = rotate_image(image, angle)

# Detect and label corners on the original image
corners_image_color_original = detect_and_label_corners(image)

# Detect and label corners on the rotated image
corners_image_color_rotated = detect_and_label_corners(rotated_image)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

# Original Image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

# Image with Harris Corners Marked and Labeled
ax2.imshow(corners_image_color_original)
ax2.set_title('Harris Corners Detected (Original)')

# Rotated Image with Harris Corners Marked and Labeled
ax3.imshow(corners_image_color_rotated)
ax3.set_title(f'Harris Corners Detected (Rotated {angle}°)')

plt.show()
