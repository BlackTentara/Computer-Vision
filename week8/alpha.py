import cv2
import numpy as np

# Load the image
image = cv2.imread('cat.jpeg')
height, width = image.shape[:2]

# Define an initial bounding box around the object you want to segment
# Format: (startX, startY, width, height)
# You can adjust this box based on your image, or use automated methods if known
rect = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

# Initialize the mask and models used by GrabCut
mask = np.zeros(image.shape[:2], np.uint8)  # Mask initialized to 0 (background)
bgd_model = np.zeros((1, 65), np.float64)   # Background model
fgd_model = np.zeros((1, 65), np.float64)   # Foreground model

# Apply the GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

# Convert the mask to a binary format: 1 (foreground), 0 (background)
binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

# Save and display the results
cv2.imwrite('generated_mask.png', binary_mask)

cv2.imshow("Original Image", image)
cv2.imshow("Generated Mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
