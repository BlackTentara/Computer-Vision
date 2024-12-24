import cv2
import numpy as np

# Load images
image1 = cv2.imread('Week6/6.jpg')
image2 = cv2.imread('Week6/5.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Use BFMatcher to match the descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography matrix to warp images
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Use homography to warp image2 onto image1
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Warp image2 to align with image1
warped_image2 = cv2.warpPerspective(image2, H, (width1 + width2, height1))

# Stitch the images together
stitched_image = np.copy(warped_image2)
stitched_image[0:height1, 0:width1] = image1

# Crop the result to remove black borders
def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y+h, x:x+w]

# Crop the black borders
final_stitched_image = crop_image(stitched_image)

# Display the final stitched image
cv2.imshow("Stitched Image", final_stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite("final_stitched_image.jpg", final_stitched_image)