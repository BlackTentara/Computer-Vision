import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'color.jpg'
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV ranges for green color
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Define the HSV ranges for yellow color
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Define the lower red range in HSV
lower_red_1 = np.array([0, 120, 70])
upper_red_1 = np.array([10, 255, 255])

# Define the upper red range in HSV
lower_red_2 = np.array([170, 120, 70])
upper_red_2 = np.array([180, 255, 255])

# Define the HSV ranges for blue color (adjusted for better accuracy)
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])

# Create masks for each color range
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
mask_red_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
mask_red_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Combine the two red masks to get a single red mask
red_mask = cv2.bitwise_or(mask_red_1, mask_red_2)

# Combine all color masks
combined_mask = cv2.bitwise_or(mask_green, mask_yellow)
combined_mask = cv2.bitwise_or(combined_mask, red_mask)
combined_mask = cv2.bitwise_or(combined_mask, mask_blue)

# Apply morphological operations to smooth the mask
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

# Apply the combined mask to the original image to get the segmented regions
segmented_image = cv2.bitwise_and(image, image, mask=combined_mask)

# Convert the images to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Display the images using Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Green, Yellow, Red, and Blue')
plt.imshow(segmented_image_rgb)
plt.axis('off')

plt.show()