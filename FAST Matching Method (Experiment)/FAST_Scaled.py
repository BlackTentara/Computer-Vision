import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def FAST_feature_matching(max_keypoints=100):
    root = os.getcwd()
    
    # Path gambar
    imgPath1 = os.path.join(root, 'week5/square2.jpg')  # Gambar pertama
    imgPath2 = os.path.join(root, 'week5/square2.jpg')  # Gambar kedua
    
    # Baca gambar dalam mode grayscale
    imgGray1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    imgGray2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)

    if imgGray1 is None or imgGray2 is None:
        print("Gambar tidak ditemukan!")
        return
    
    # Flip gambar kedua secara vertikal
    imgGray2 = cv.flip(imgGray2, 0)  # 1 untuk horizontal, 0 untuk vertikal

    # Scaling gambar kedua 1.5x
    scale_factor = 1.5
    imgGray2 = cv.resize(imgGray2, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)

    # Inisialisasi detektor FAST
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(10)
    
    # Deteksi keypoints menggunakan FAST
    keypoints1 = fast.detect(imgGray1, None)
    keypoints2 = fast.detect(imgGray2, None)

    # Urutkan keypoints berdasarkan respons dan batasi ke max_keypoints
    keypoints1 = sorted(keypoints1, key=lambda x: x.response, reverse=True)[:max_keypoints]
    keypoints2 = sorted(keypoints2, key=lambda x: x.response, reverse=True)[:max_keypoints]

    # Extract the keypoint locations (pt) for both images
    points1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
    points2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

    # Compare the locations of the keypoints using Euclidean distance
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(points1, points2)

    # Urutkan berdasarkan jarak
    matches = sorted(matches, key=lambda x: x.distance)

    # Gambar kecocokan di antara kedua gambar
    img_matches = cv.drawMatches(imgGray1, keypoints1, imgGray2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Menampilkan hasil kecocokan
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches, cmap='gray')  # Gunakan img_matches untuk menampilkan hasil matching
    plt.title("FAST Keypoints Matching (Strictly Keypoint Location Comparison)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    FAST_feature_matching(max_keypoints=100)
