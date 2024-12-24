import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_object_and_merge_with_grabcut(image_path, new_bg_path, rect=None):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a mask with the same dimensions as the image, initialized to '0'
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # If no rectangle is provided, create a default one
    if rect is None:
        rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to separate foreground and background
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image_rgb * mask2[:, :, np.newaxis]

    # Load the new background image
    new_bg = cv2.imread(new_bg_path)
    new_bg = cv2.cvtColor(new_bg, cv2.COLOR_BGR2RGB)
    new_bg_resized = cv2.resize(new_bg, (image.shape[1], image.shape[0]))

    # Create the final image by combining the foreground with the new background
    combined_image = new_bg_resized * (1 - mask2[:, :, np.newaxis]) + foreground

    # Display the images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(new_bg_resized)
    plt.title('New Background Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(combined_image.astype(np.uint8))
    plt.title('Combined Image')
    plt.axis('off')

    plt.show()

    # Save the combined image
    cv2.imwrite("combined_output_grabcut.png", cv2.cvtColor(combined_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

# Paths to the images
image_path = 'cat.jpeg'  # Path to the foreground image
new_bg_path = 'forest.jpg'  # Path to the new background image

# Run the function
extract_object_and_merge_with_grabcut(image_path, new_bg_path)
