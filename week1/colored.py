import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the RGB image
image = mpimg.imread('week1/chick.jpeg')

# Check if the image is indeed RGB
if image.ndim == 3 and image.shape[2] == 3:
    # Separate the RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]



    # Copy the red channel for modification
    modified_red_channel = np.copy(red_channel)

    # Identify pixels where red value > 150 and add 50 to the red value
    mask = red_channel > 150
    modified_red_channel[mask] = np.clip(red_channel[mask] -50, 0, 255)

    # Reassemble the image with the modified red channel
    modified_image = np.stack([modified_red_channel, green_channel, blue_channel], axis=2)

    # Plot the original and modified images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax1.imshow(image)
    ax1.set_title('Original Image')

    # Display the modified image
    ax2.imshow(modified_image)
    ax2.set_title('Modified Image')

    plt.show()
else:
    print("The image is not in RGB format.")
