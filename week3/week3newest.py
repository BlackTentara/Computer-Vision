import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import convolve
from numpy.fft import fft2, ifft2, fftshift, ifftshift

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
            intensity = binary_sobel_image_x[y, x]
            ax3.set_title(f'Sobel Horizontal - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax4:
            intensity = binary_sobel_image_y[y, x]
            ax4.set_title(f'Sobel Vertical - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax5:
            intensity = sharpened_image[y, x]
            ax5.set_title(f'Sharpened Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax6:
            intensity = restored_image[y, x]
            ax6.set_title(f'Restored Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax7:
            intensity = binary_sobel_image[y, x]
            ax7.set_title(f'Sobel Image - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax8:
            intensity = fourier_low_pass_image[y, x]
            ax8.set_title(f'Fourier Low-Pass - Intensity at ({x}, {y}): {intensity:.2f}')
        elif event.inaxes == ax9:
            intensity = fourier_high_pass_image[y, x]
            ax9.set_title(f'Fourier High-Pass - Intensity at ({x}, {y}): {intensity:.2f}')
        fig.canvas.draw_idle()

# Load the grayscale image
image = mpimg.imread('week2/cat.jpg')

# Convert to grayscale if the image is in RGB
if len(image.shape) == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Normalize the image to the 0-255 range if necessary
if image.max() <= 1.0:
    image = (image * 255).astype(np.uint8)

# Gaussian blur kernel (5x5 kernel)
gaussian_kernel = np.array([
    [1,  4,  7,  10,  7,  4,  1],
    [4, 16, 26,  33, 26, 16,  4],
    [7, 26, 41,  52, 41, 26,  7],
    [10, 33, 52, 64, 52, 33, 10],
    [7, 26, 41,  52, 41, 26,  7],
    [4, 16, 26,  33, 26, 16,  4],
    [1,  4,  7,  10,  7,  4,  1]
]) / 1003.0

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
blurred_image = convolve(image, gaussian_kernel)

# Apply Sobel edge detection
sobel_image_x = convolve(image, sobel_kernel_x)
sobel_image_y = convolve(image, sobel_kernel_y)
sobel_image = np.hypot(sobel_image_x, sobel_image_y)
sobel_image = np.clip(sobel_image, 0, 255)

# Apply binary threshold to Sobel X and Y images
threshold = 64  # Adjusting threshold to a lower value
binary_sobel_image_x = np.where(sobel_image_x > threshold, 255, 0).astype(np.uint8)
binary_sobel_image_y = np.where(sobel_image_y > threshold, 255, 0).astype(np.uint8)
binary_sobel_image = np.where(sobel_image > threshold, 255, 0).astype(np.uint8)

# Apply sharpening filter and clip the values to [0, 255]
sharpened_image = convolve(image, sharpening_kernel)
sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

# Apply restoration (Gaussian blur followed by sharpening)
restored_image = convolve(blurred_image, sharpening_kernel)
restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)

# Fourier transform
f_transform = fftshift(fft2(image))

# Create separate radii for low-pass and high-pass filters
r_low = 10   # Adjust the radius for the low-pass filter
r_high = 1  # Adjust the radius for the high-pass filter

# Create masks for low-pass and high-pass filters
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2

# Low-pass mask
low_pass_mask = np.zeros((rows, cols), np.uint8)
x, y = np.ogrid[:rows, :cols]
low_pass_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r_low ** 2
low_pass_mask[low_pass_area] = 1

# High-pass mask
high_pass_mask = np.ones((rows, cols), np.uint8)
high_pass_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r_high ** 2
high_pass_mask[high_pass_area] = 0

# Apply low-pass filter
f_low_pass = f_transform * low_pass_mask
f_low_pass_ishift = ifftshift(f_low_pass)
fourier_low_pass_image = np.abs(ifft2(f_low_pass_ishift))
fourier_low_pass_image = np.clip(fourier_low_pass_image, 0, 255).astype(np.uint8)

# Apply high-pass filter
f_high_pass = f_transform * high_pass_mask
f_high_pass_ishift = ifftshift(f_high_pass)
fourier_high_pass_image = np.abs(ifft2(f_high_pass_ishift))
fourier_high_pass_image = np.clip(fourier_high_pass_image, 0, 255).astype(np.uint8)

# Create subplots to display the images, now with 4 rows and 3 columns (for 10 subplots)
fig, axs = plt.subplots(3, 4, figsize=(18, 24))  # 4 rows, 3 columns to accommodate ax10

# Assign each axis to a variable (taking only 10 axes)
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axs.flatten()[:9]
for ax in axs.flatten()[9:]:
    ax.remove()
   

# Display the original image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

# Display the blurred image
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')

# Display the binary Sobel Horizontal (X) edge detection image
ax3.imshow(binary_sobel_image_x, cmap='gray')
ax3.set_title('Sobel Horizontal (X)')

# Display the binary Sobel Vertical (Y) edge detection image
ax4.imshow(binary_sobel_image_y, cmap='gray')
ax4.set_title('Sobel Vertical (Y)')

# Display the sharpened image
ax5.imshow(sharpened_image, cmap='gray')
ax5.set_title('Sharpened Image')

# Display the restored image
ax6.imshow(restored_image, cmap='gray')
ax6.set_title('Restored Image')

# Display the binary Sobel magnitude edge detection
ax7.imshow(binary_sobel_image, cmap='gray')
ax7.set_title('Sobel Edge Detection')

# Display the Fourier low-pass filtered image
ax8.imshow(fourier_low_pass_image, cmap='gray')
ax8.set_title('Fourier Low-Pass Filter')

# Display the Fourier high-pass filtered image
ax9.imshow(fourier_high_pass_image, cmap='gray')
ax9.set_title('Fourier High-Pass Filter')



# Connect the mouse movement event to the handler
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()
