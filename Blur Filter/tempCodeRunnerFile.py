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
            intensity = blurred_image[y, x]  # Blurred image intensity
            ax2.set_title(f'Blurred Image - Intensity at ({x}, {y}): {intensity:.2f}')
        fig.canvas.draw_idle()

# Load the grayscale image
image = mpimg.imread('week2/cat.jpg')

# Check if the image is indeed grayscale and adjust the intensity range
if len(image.shape) == 3:  # If the image has multiple channels
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

# Normalize the image to the 0-255 range if it's not already
if image.max() <= 1.0:  # If the image is in the 0-1 range, scale it up
    image = (image * 255).astype(np.uint8)

# Initialize the blurred image array
height, width = image.shape
blurred_image = np.copy(image)

# Define a larger blurring kernel (7x7 box blur)
kernel_size = 15
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

def apply_blur_kernel(image, x, y, kernel):
    kernel_size = kernel.shape[0]
    k_half = kernel_size // 2
    pixel_sum = 0.0
    
    for ky in range(-k_half, k_half + 1):
        for kx in range(-k_half, k_half + 1):
            ny, nx = y + ky, x + kx
            if 0 <= ny < height and 0 <= nx < width:
                pixel_sum += image[ny, nx] * kernel[ky + k_half, kx + k_half]
    
    return pixel_sum

# Apply blur to specific regions
for y in range(height):
    for x in range(width):
        if x > width // 4 and x < 3 * width // 4 and y > height // 4 and y < 3 * height // 4:
            blurred_image[y, x] = apply_blur_kernel(image, x, y, kernel)

# Create subplots to display both images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

# Display the blurred image
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')

# Connect the mouse movement event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
