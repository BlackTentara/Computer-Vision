import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter, sobel, convolve

# Function to handle mouse motion
def on_mouse_move(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        intensity = None
        if event.inaxes == ax1:
            intensity = image[y, x]
            ax1.set_title(f'Original Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax2:
            intensity = blurred_image[y, x]
            ax2.set_title(f'Blurred Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax3:
            intensity = sobel_image[y, x]
            ax3.set_title(f'Sobel Edge Detection - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax4:
            intensity = sharpened_image[y, x]
            ax4.set_title(f'Sharpened Image - Intensity at ({x}, {y}): {intensity:.2f}')
        fig.canvas.draw_idle()

# Load the grayscale image
image = mpimg.imread('week2/cat.jpg')

# Convert to grayscale if the image is in RGB
if len(image.shape) == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Normalize the image to the 0-255 range
if image.max() <= 1.0:
    image = (image * 255).astype(np.uint8)

# Apply Gaussian blur
blurred_image = gaussian_filter(image, sigma=3)

# Apply Sobel edge detection
sobel_image_x = sobel(image, axis=0)
sobel_image_y = sobel(image, axis=1)
sobel_image = np.hypot(sobel_image_x, sobel_image_y)

# Apply sharpening filter
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened_image = convolve(image, sharpening_kernel)

# Clip the sharpened image to be within the valid range [0, 255]
sharpened_image = np.clip(sharpened_image, 0, 255)

# Create subplots to display the images
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

# Display the original image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

# Display the blurred image
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')

# Display the Sobel edge detection image
ax3.imshow(sobel_image, cmap='gray')
ax3.set_title('Sobel Edge Detection')

# Display the sharpened image
ax4.imshow(sharpened_image, cmap='gray')
ax4.set_title('Sharpened Image')

# Connect the mouse movement event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
