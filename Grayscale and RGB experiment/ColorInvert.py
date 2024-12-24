import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to handle mouse motion
def on_mouse_move(event):
    if event.inaxes:  # Check if the mouse is within the axes
        x, y = int(event.xdata), int(event.ydata)
        if event.inaxes == ax1:
            intensity = image[y, x]  # Original image intensity
            ax1.set_title(f'Original Image - Intensity at ({x}, {y}): {intensity}')
        elif event.inaxes == ax2:
            intensity = modified_image[y, x]  # Modified image intensity
            ax2.set_title(f'Modified Image - Intensity at ({x}, {y}): {intensity}')
        fig.canvas.draw_idle()

# Load the color image
image = mpimg.imread('week1/chick.jpeg')

# Handle images with an alpha channel
if image.shape[2] == 4:  # Check if the image has 4 channels (including alpha)
    image = image[..., :3]  # Exclude the alpha channel

# Ensure the image is in the 0-255 range
if image.max() <= 1.0:  # If the image is in the 0-1 range, scale it up
    image = (image * 255).astype(np.uint8)

# Initialize the modified image array (same size as original)
height, width, channels = image.shape
modified_image = np.copy(image)

# Adjust color channels based on conditions
for y in range(height):
    for x in range(width):
        r, g, b  = image[y, x]
        if x < width / 2:
            # Invert colors on the left half of the image
            modified_image[y, x] = [255 - r, 255 - g, 255 - b]
            
        else:
            # Flip colors on the right half of the image (swap red and blue)
            modified_image[y, x] = [r,g,b]

# Create subplots to display both images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax1.imshow(image)
ax1.set_title('Original Image')

# Display the modified image
ax2.imshow(modified_image)
ax2.set_title('Modified Image')

# Connect the mouse movement event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
