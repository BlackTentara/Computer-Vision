import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_object_extraction_with_grabcut(foreground_path, background_path, rect=None, iterations=5):
    # Load the foreground and background images
    foreground = cv2.imread(foreground_path)
    background = cv2.imread(background_path)

    # Resize the background to match the foreground dimensions
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # Define the rectangular region for GrabCut if not provided
    if rect is None:
        rect = (15, 15, foreground.shape[1] - 30, foreground.shape[0] - 30)  # Adjusted for better edge coverage

    # Initialize mask and models for GrabCut
    mask = np.zeros(foreground.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Apply initial GrabCut with the provided rectangle
    cv2.grabCut(foreground, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Refine mask to include only the definite foreground
    mask_final = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    extracted_object = cv2.bitwise_and(foreground, foreground, mask=mask_final)

    # Create an inverse mask to blend with the background
    mask_inv = cv2.bitwise_not(mask_final * 255)
    background_area = cv2.bitwise_and(background, background, mask=mask_inv)

    # Combine the extracted object with the background
    final_output = cv2.add(background_area, extracted_object)

    # Display results with matplotlib
    plt.figure(figsize=(15, 5))

    # Show the original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Show the extracted object
    plt.subplot(1, 3, 2)
    plt.title("Extracted Object")
    plt.imshow(cv2.cvtColor(extracted_object, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Show the combined output image
    plt.subplot(1, 3, 3)
    plt.title("Output Image")
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
    
    # Save the final output image
    cv2.imwrite("output.png", final_output)

# Paths to the foreground and background images
foreground_path = "pug.jpg"
background_path = "forest.jpg"

# Call the function with refined GrabCut processing
refine_object_extraction_with_grabcut(foreground_path, background_path)
