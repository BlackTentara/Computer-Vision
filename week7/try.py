import cv2
import numpy as np
import glob
import imutils

# Path gambar dan output
image_dir = 'week7/imagesNew/*.jpg'
output_path = "StitchedOutput_Manual_SIFT_Final.png"

# Fungsi untuk resize gambar agar proses lebih cepat
def resize_image(image, max_width=800):
    if image.shape[1] > max_width:
        return imutils.resize(image, width=max_width)
    return image

# Fungsi untuk mencocokkan keypoints antara dua gambar dan menghitung homografi
def match_keypoints(keypointsA, keypointsB, descriptorsA, descriptorsB, ratio=0.75, reprojThresh=4.0):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = matcher.knnMatch(descriptorsA, descriptorsB, 2)
    matches = []
    for m, n in rawMatches:
        if m.distance < ratio * n.distance:
            matches.append(m)
    if len(matches) > 4:
        ptsA = np.float32([keypointsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([keypointsB[m.trainIdx].pt for m in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return matches, H, status
    else:
        return None, None, None

# Fungsi untuk stitching dua gambar
def stitch_images(imageA, imageB):
    sift = cv2.SIFT_create()
    keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)
    matches, H, status = match_keypoints(keypointsA, keypointsB, descriptorsA, descriptorsB)
    if H is None:
        print("Homography tidak ditemukan. Tidak dapat melakukan stitching.")
        return None
    heightA, widthA = imageA.shape[:2]
    heightB, widthB = imageB.shape[:2]
    result = cv2.warpPerspective(imageB, H, (widthA + widthB, max(heightA, heightB)))
    result[0:heightA, 0:widthA] = imageA
    return result

# Fungsi untuk blending transisi antar gambar
def blend_images(imageA, warped_imageB):
    # Sesuaikan ukuran gambar A dan hasil warp B agar memiliki ukuran yang sama
    heightA, widthA = imageA.shape[:2]
    heightB, widthB = warped_imageB.shape[:2]

    # Ambil ukuran minimal dari dua gambar
    min_height = min(heightA, heightB)
    min_width = min(widthA, widthB)

    # Potong kedua gambar ke ukuran minimal
    cropped_imageA = imageA[:min_height, :min_width]
    cropped_warped_imageB = warped_imageB[:min_height, :min_width]

    # Simple averaging blend untuk menggabungkan gambar
    blend_result = np.where(cropped_warped_imageB > 0, cropped_warped_imageB, cropped_imageA)
    return blend_result

# Mengambil semua gambar dari direktori
image_paths = glob.glob(image_dir)
if len(image_paths) < 2:
    print(f"Error: Ditemukan kurang dari dua gambar untuk stitching.")
    exit()

# Membaca dan memproses gambar
images = []
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membuka gambar {image_path}.")
        continue
    img = resize_image(img)
    images.append(img)

# Mulai proses stitching dari gambar pertama
stitched_image = images[0]

for i in range(1, len(images)):
    print(f"Stitching gambar {i+1} dari {len(images)}...")
    warped_image = stitch_images(stitched_image, images[i])
    if warped_image is None:
        print("Proses stitching gagal.")
        exit()

    # Blending gambar hasil warp dengan gambar sebelumnya
    stitched_image = blend_images(stitched_image, warped_image)

# Simpan hasil stitched image
if stitched_image is not None:
    cv2.imwrite(output_path, stitched_image)
    print(f"Stitched image berhasil disimpan di {output_path}.")
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
