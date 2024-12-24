import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the previous color image (foreground with matte):
I = cv2.imread('matting1.jpg')  # Load the image with foreground
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)/255  # Convert to RGB and normalize to range [0, 1]
n_rows = I.shape[0]  # Number of rows in the image
n_cols = I.shape[1]  # Number of columns in the image
n_pixels = n_rows * n_cols  # Total number of pixels in the image
I = np.reshape(I, (n_pixels, 3))  # Reshape the image into a list of pixels

# Load exact matte (alpha):
alpha_ex = cv2.imread('matting1.jpg', cv2.IMREAD_GRAYSCALE)/255  # Ground truth alpha matte
alpha_ex = np.reshape(alpha_ex, (n_pixels, 1))  # Reshape to a list of pixels (n_pixels, 1)

# Generate a version with a green screen background:
G_B = 1  # Green background intensity
I_green_bg = alpha_ex * I + (1 - alpha_ex) * [0, G_B, 0]  # Apply green screen background
I_green_bg = np.reshape(I_green_bg, (n_rows, n_cols, 3))  # Reshape back to original dimensions
plt.imshow(I_green_bg), plt.axis('off'), plt.show()  # Show the image with the green screen background

# Extract the RGB components of the image:
R_I = I_green_bg[:, :, 0]  # Red channel
G_I = I_green_bg[:, :, 1]  # Green channel
B_I = I_green_bg[:, :, 2]  # Blue channel

# Coefficients for the alpha calculation:
a_0 = 0  # Coefficient for the constant term
a_1 = 0.5  # Coefficient for the green channel
a_2 = 0  # Coefficient for the red channel

# Compute the alpha matte with the given formula:
alpha = (B_I - a_1*(G_I - G_B) - a_2*R_I) / (a_0 + a_1*G_B)  # Compute the estimated alpha
alpha = np.clip(alpha, 0, 1)  # Clamp alpha values to [0, 1]
plt.imshow(alpha, cmap='gray'), plt.axis('off'), plt.show()  # Show the estimated alpha matte

# Compute the relative error with the exact alpha matte:
alpha = np.reshape(alpha, (n_pixels, 1))  # Reshape the estimated alpha to match pixel format
error = np.linalg.norm(alpha - alpha_ex) / np.linalg.norm(alpha_ex)  # Compute relative error
print(f'Error (alpha): {error:.2e}')  # Print the error

# Load a new background image (K):
K = cv2.imread('kor.jpeg')  # Load a background image
K = cv2.cvtColor(K, cv2.COLOR_BGR2RGB)/255  # Convert to RGB and normalize to range [0, 1]
K_resized = cv2.resize(K, (n_cols, n_rows))  # Resize the background to match foreground dimensions
K = np.reshape(K_resized, (n_pixels, 3))  # Reshape the background image into a list of pixels

# Compute the new image J with the background K:
alpha = np.reshape(alpha, (n_pixels, 1))  # Ensure alpha is reshaped
I = np.reshape(I, (n_pixels, 3))  # Reshape foreground image

# Combine the foreground with the new background using the alpha matte
J = alpha * I + (1 - alpha) * K
J = np.clip(J, 0, 1)  # Clamp values to [0, 1]
J = np.reshape(J, (n_rows, n_cols, 3))  # Reshape the image back to original dimensions

# Display the final composited image with the new background
plt.imshow(J), plt.axis('off'), plt.show()
