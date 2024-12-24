import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to handle mouse motion
def on_mouse_move(event):
    if event.inaxes:  # Check if the mouse is within the axes
        x, y = int(event.xdata), int(event.ydata)
        intensity = image[y, x]
        ax.set_title(f'Intensity at ({x:.0f}, {y:.0f}): {intensity:.2f}')
        fig.canvas.draw_idle()

# Load the grayscale image
image = mpimg.imread('week1/greyscale.png')

# Check if the image is indeed grayscale and adjust the intensity range
if len(image.shape) == 3:  # If the image has multiple channels
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

# Normalize the image to the 0-255 range if it's not already
if image.max() <= 1.0:  # If the image is in the 0-1 range, scale it up
    image = image * 255

# Plot the image
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.set_title('Move the mouse over the image to see intensity')

# Connect the motion_notify_event to the on_mouse_move function
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

array =[]
height, width = image.shape
for y in range(1):
        for x in range(width):
            intensity = image[y,x]
            
            array.append([x,y,intensity])
# Display the plot

print(array)
plt.show()






