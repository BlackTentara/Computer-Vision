import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to segment a chosen color
def segment_color(image, color_choice):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for each color
    color_ranges = {
        "green": (np.array([35, 50, 50]), np.array([85, 255, 255])),
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ],
        "blue": (np.array([90, 100, 100]), np.array([130, 255, 255]))
    }

    # Validate user color choice
    if color_choice not in color_ranges:
        raise ValueError(f"Color '{color_choice}' not supported. Choose from: {list(color_ranges.keys())}")

    # Create a mask for the selected color
    if color_choice == "red":
        # Red requires two ranges, combine both masks
        mask1 = cv2.inRange(hsv_image, *color_ranges["red"][0])
        mask2 = cv2.inRange(hsv_image, *color_ranges["red"][1])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Use a single range for other colors
        lower, upper = color_ranges[color_choice]
        mask = cv2.inRange(hsv_image, lower, upper)

    # Apply morphological operations to smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image to get the segmented region
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert images to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Segmented {color_choice.capitalize()} Color')
    plt.imshow(segmented_image_rgb)
    plt.axis('off')

    plt.show()

# Load the image
image_path = 'color.jpg'
image = cv2.imread(image_path)

# Get user input for the color choice
color_choice = input("Enter the color to segment (red, green, blue, yellow): ").strip().lower()

# Segment the chosen color
segment_color(image, color_choice)
