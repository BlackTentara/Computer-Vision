import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def ORB_feature_matching(max_keypoints=100):
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

    # Use BFMatcher with Hamming distance and enable crossCheck
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Find homography using RANSAC
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Draw only inliers
    draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers
                       flags=2)

    img_matches = cv.drawMatches(imgGray1, keypoints1, imgGray2, keypoints2, matches, None, **draw_params)

    # Display the matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.show()

if __name__ == '__main__':
    ORB_feature_matching(max_keypoints=100)
