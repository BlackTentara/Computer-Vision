from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
ap.add_argument("-m", "--max_width", type=int, default=5000,
                help="maximum allowable width of the final stitched image")
args = vars(ap.parse_args())

# Grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# Calculate the resize width dynamically based on the number of images
num_images = len(imagePaths)
resize_width = args["max_width"] // num_images

print(f"[INFO] Resizing images to {resize_width}px width")

# Resize and load images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # Resize the image to the calculated width (keeping aspect ratio)
    image = imutils.resize(image, width=resize_width)
    images.append(image)

# Function to attempt stitching with different modes
def stitch_images(images):
    print("[INFO] stitching images with default mode...")
    stitcher = cv2.Stitcher_create()

    # First try the default mode
    (status, stitched) = stitcher.stitch(images)

    # If stitching fails, try another mode (PANORAMA vs SCANS)
    if status != 0:
        print("[INFO] default stitching failed, trying with scan mode...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        (status, stitched) = stitcher.stitch(images)

    return status, stitched

# Perform image stitching
status, stitched = stitch_images(images)

# Check if the stitching was successful
if status == 0:
    # Optionally expand the canvas if you want more space around the stitched image
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # Write the output stitched image to disk
    cv2.imwrite(args["output"], stitched)
    print(f"[INFO] stitched image saved as {args['output']}")

    # Display the output stitched image
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
else:
    print(f"[INFO] image stitching failed with status code {status}.")

    # Create a larger blank canvas to fit all images if stitching fails
    print("[INFO] Creating a canvas to fit all images...")
    total_width = sum(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Place all images on the canvas
    x_offset = 0
    for image in images:
        canvas[:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1]

    # Save the full-resolution image on the canvas
    cv2.imwrite(args["output"], canvas)
    print(f"[INFO] Images placed on canvas saved as {args['output']}")

    # Display the canvas with all images
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)
