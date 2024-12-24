import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def ORB_feature_matching_with_ratio_test(max_keypoints=500, ratio_threshold=0.75):
    # Load the image
    root = os.getcwd()
    imgPath1 = os.path.join(root, 'week5/cat.jpg')
    imgPath2 = os.path.join(root, 'week5/cat.jpg')
    imgGray1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGray2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    # Flip the second image vertically
    imgGray2 = cv.flip(imgGray2, 0)

    # Detect keypoints and compute descriptors using ORB
    orb = cv.ORB_create(max_keypoints)
    keypoints1, descriptors1 = orb.detectAndCompute(imgGray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imgGray2, None)

    # Use BFMatcher with Hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Find homography using RANSAC
    if len(good_matches) > 4:  # Minimum needed for homography
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers if mask is available
                       flags=2)

    img_matches = cv.drawMatches(imgGray1, keypoints1, imgGray2, keypoints2, good_matches, None, **draw_params)

    # Display the matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.show()

if __name__ == '__main__':
    ORB_feature_matching_with_ratio_test(max_keypoints=500, ratio_threshold=0.75)
