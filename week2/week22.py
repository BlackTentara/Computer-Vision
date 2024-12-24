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
            ax1.set_title(f'Original Image - Intensity at ({x}, {y}): {intensity}')
        elif event.inaxes == ax2:
            intensity = blurred_image[y, x]  # Blurred image intensity
            ax2.set_title(f'Blurred Image - Intensity at ({x}, {y}): {intensity}')
        fig.canvas.draw_idle()

# Load the RGB image
image = mpimg.imread('week2/cat color.jpeg')

# Handle images with an alpha channel
if image.shape[2] == 4:  # Check if the image has 4 channels (including alpha)
    image = image[..., :3]  # Exclude the alpha channel

# Normalize the image to the 0-255 range if it's not already
if image.max() <= 1.0:  # If the image is in the 0-1 range, scale it up
    image = (image * 255).astype(np.uint8)

# Initialize the blurred image array (same size as original)
height, width, channels = image.shape
blurred_image = np.copy(image)

# Define a manual blurring kernel (3x3 matrix example)
kernel = np.array([[1, 1, 1,1,1],
                   [1, 1, 1,1,1],
                   [1, 1, 1,1,1],
                   [1, 1, 1,1,1],
                   [1, 1, 1,1,1],]) / 25.0

def apply_blur_kernel(image, x, y, kernel):
    kernel_size = kernel.shape[0]
    k_half = kernel_size // 2
    pixel_sum = np.zeros(3)  # For RGB channels
    
    for ky in range(-k_half, k_half + 1):
        for kx in range(-k_half, k_half + 1):
            ny, nx = y + ky, x + kx
            if 0 <= ny < height and 0 <= nx < width:
                pixel_sum += image[ny, nx] * kernel[ky + k_half, kx + k_half]
    
    return pixel_sum


sharp_kernel = np.array([ 
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -1, -1,  0],
    [-1, -1, 25, -1, -1],
    [ 0, -1, -1, -1,  0],
    [ 0,  0, -1,  0,  0],])
def apply_sharp_kernel(image, x, y, sharp_kernel):
    sharp_kernel_size = sharp_kernel.shape[0]
    k_half = sharp_kernel_size // 2
    pixel_sum = np.zeros(3)  # For RGB channels
    
    for ky in range(-k_half, k_half + 1):
        for kx in range(-k_half, k_half + 1):
            ny, nx = y + ky, x + kx
            if 0 <= ny < height and 0 <= nx < width:
                pixel_sum += image[ny, nx] * sharp_kernel[ky + k_half, kx + k_half]
    
    return pixel_sum


# Apply blur to specific regions
for y in range(height):
    for x in range(width):
        if x > width // 4 and x < 3 * width // 4 and y > height // 4 and y < 3 * height // 4:
            blurred_image[y, x] = apply_blur_kernel(image, x, y, kernel)
        else:
            blurred_image[y, x] = apply_sharp_kernel(image, x, y, sharp_kernel)

# Create subplots to display both images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax1.imshow(image)
ax1.set_title('Original Image')

# Display the blurred image
ax2.imshow(blurred_image)
ax2.set_title('Blurred Image')

# Connect the mouse movement event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
