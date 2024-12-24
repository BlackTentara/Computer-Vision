import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def FAST():
    root = os.getcwd()
    imgPath1 = os.path.join(root, 'week5/cat.jpg')
    imgPath2 = os.path.join(root, 'week5/cat.jpg')  # Add the second image path
    imgGray1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGray2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    fast = cv.FastFeatureDetector_create()
    minIntensityDiff = 75
    fast.setThreshold(minIntensityDiff)
    keypoints1 = fast.detect(imgGray1)
    keypoints2 = fast.detect(imgGray2)

    # Convert keypoints to numpy arrays for matching
    keypoints1_np = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
    keypoints2_np = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

    # Use BFMatcher to find matches between keypoints
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    descriptors1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32).reshape(-1, 2)
    descriptors2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32).reshape(-1, 2)
    matches = bf.match(descriptors1, descriptors2)

    # Draw matches
    img_matches = cv.drawMatches(imgGray1, keypoints1, imgGray2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Scale up the size of each keypoint
    scale_factor = 1  # Increase this value for larger circles
    for kp in keypoints1:
        kp.size *= scale_factor  # Enlarge each keypoint's size
    for kp in keypoints2:
        kp.size *= scale_factor  # Enlarge each keypoint's size

    # Draw the enlarged keypoints
    imgGray1 = cv.drawKeypoints(imgGray1, keypoints1, imgGray1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgGray2 = cv.drawKeypoints(imgGray2, keypoints2, imgGray2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Combine the images side by side
    combined_img = np.hstack((imgGray1, imgGray2))

    plt.figure()
    plt.imshow(img_matches)  # Use img_matches to display matches
    plt.show()

if __name__ == '__main__':
    FAST()
