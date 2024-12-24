import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

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

# Save the images to the local directory
output_dir = 'filtered_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save images
plt.imsave(f'{output_dir}/original_image.png', image, cmap='gray')
plt.imsave(f'{output_dir}/blurred_image.png', blurred_image, cmap='gray')
plt.imsave(f'{output_dir}/sobel_image.png', sobel_image, cmap='gray')
plt.imsave(f'{output_dir}/sharpened_image.png', sharpened_image, cmap='gray')

print(f"Images saved in {output_dir} directory.")


