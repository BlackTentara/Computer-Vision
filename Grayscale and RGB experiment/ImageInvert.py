import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to handle mouse motion
def on_mouse_move(event):
    if event.inaxes:  # Check if the mouse is within the axes
        x, y = int(event.xdata), int(event.ydata)
        intensity = None
        if event.inaxes == ax1:
            intensity = image[y, x]  # Original image intensity
            ax1.set_title(f'Original Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax2:
            intensity = inverted_image[y, x]  # Inverted image intensity
            ax2.set_title(f'Inverted Image - Intensity at ({x}, {y}): {intensity:.2f}')
        fig.canvas.draw_idle()

# Load the grayscale image
image = mpimg.imread('week1/bird.jpg')

# Check if the image is indeed grayscale and adjust the intensity range
if len(image.shape) == 3:  # If the image has multiple channels
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

# Normalize the image to the 0-255 range if it's not already
if image.max() <= 1.0:  # If the image is in the 0-1 range, scale it up
    image = image * 255

# Initialize the inverted image array
height, width = image.shape
inverted_image = np.zeros_like(image)

# Manually invert the grayscale colors
for y in range(height):
    for x in range(width):
        if (x<width/2):
           
            inverted_image[y, x] = 255- image[height - 1 -y, x]
        else:
            inverted_image[y, x] =  image[y, width - 1 - x]


# Create subplots to display both images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

# Display the inverted image
ax2.imshow(inverted_image, cmap='gray')
ax2.set_title('Inverted Image')

plt.show()

# Manually flip the image horizontally
# flipped_horizontal = np.copy(image)
# for y in range(height):
#     for x in range(width):
#         flipped_horizontal[y, x] = image[y, width - 1 - x]

# # Manually flip the image vertically
# flipped_vertical = np.copy(image)
# for y in range(height):
#     for x in range(width):
#         flipped_vertical[y, x] = image[height - 1 - y, x]