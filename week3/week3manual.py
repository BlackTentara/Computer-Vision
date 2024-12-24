import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Function to apply a filter kernel to an image
def apply_filter(image, kernel):
    kernel_size = kernel.shape[0]
    k_half = kernel_size // 2
    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            pixel_sum = 0.0
            for ky in range(-k_half, k_half + 1):
                for kx in range(-k_half, k_half + 1):
                    ny, nx = y + ky, x + kx
                    if 0 <= ny < height and 0 <= nx < width:
                        pixel_sum += image[ny, nx] * kernel[ky + k_half, kx + k_half]
            filtered_image[y, x] = pixel_sum
            
    return np.clip(filtered_image, 0, 255)

# Load the grayscale image
image = mpimg.imread('week2/cat.jpg')

# Convert to grayscale if the image is in RGB
if len(image.shape) == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Normalize the image to the 0-255 range
if image.max() <= 1.0:
    image = (image * 255).astype(np.uint8)

# Gaussian blur kernel (7x7 kernel with sigma = 1)
gaussian_kernel = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
]) / 273.0

# Sobel kernels for edge detection
sobel_kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_kernel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Sharpening kernel
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# Apply Gaussian blur
blurred_image = apply_filter(image, gaussian_kernel)

# Apply Sobel edge detection
sobel_image_x = apply_filter(image, sobel_kernel_x)
sobel_image_y = apply_filter(image, sobel_kernel_y)
sobel_image = np.hypot(sobel_image_x, sobel_image_y)
sobel_image = np.clip(sobel_image, 0, 255)

# Apply sharpening filter
sharpened_image = apply_filter(image, sharpening_kernel)

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

