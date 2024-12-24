import cv2
import glob
import imutils
import numpy as np

# deklarasi image directory dan path output
image_dir = 'week7/Uts3/*.jpg' 
output_path = "StitchedOutputNewLast.png"

# Kumpulkan semua image directory yang dibutuhkan dalam stitching
image_paths = glob.glob(image_dir)

# Periksa di folder/path tersebut ada minimal 2 image
if len(image_paths) < 2:
    print(f"Error: Need at least two images for stitching, but found {len(image_paths)}.")
    exit()

# Array untuk semua image
images = []

# Resize Image
def resize_image(image, max_width=800):
    # Resize image dengan mempertahankan aspek ratio
    if image.shape[1] > max_width:
        return imutils.resize(image, width=max_width)
    return image

# Load semua image dan preprocessing
for image_path in image_paths:
    img = cv2.imread(image_path)
    
    # Error handling untuk load image
    if img is None:
        print(f"Error loading image {image_path}. Skipping this file.")
        continue
    
    # Resize image
    img = resize_image(img)
    
    # Tambah image dalam array
    images.append(img)
    
cv2.destroyAllWindows()

# Periksa Image harus lebih dari 2
if len(images) < 2:
    print("Error: Insufficient valid images for stitching.")
    exit()

# Inisialisasi OpenCV Stitcher
imageStitcher = cv2.Stitcher_create()

# Stitch gambar
print("Stitching images...")
error, stitched_img = imageStitcher.stitch(images)

# Sukses
if error == cv2.Stitcher_OK:
    
    print("Stitching successful!")
    
    # Save Hasil Stitching
    cv2.imwrite(output_path, stitched_img)
    print(f"Stitched image saved at {output_path}")
    
    # Cari area hitam dengan RGB value 0,0,0
    mask = cv2.inRange(stitched_img, np.array([0, 0, 0]), np.array([0, 0, 0]))

    # Jika ada area hitam
    if np.any(mask > 0):
        print("Black areas detected, applying inpainting to fill gaps...")

        # Mewarnai black pixel dengan warna area sekitar
        stitched_img_inpainted = cv2.inpaint(stitched_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Save 
        cv2.imwrite(output_path.replace(".png", "_inpainted.png"), stitched_img_inpainted)
        print(f"Inpainted stitched image saved at {output_path.replace('.png', '_inpainted.png')}")

        # Display 
        cv2.imshow("Inpainted Stitched Image", stitched_img_inpainted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No black areas detected, no inpainting required.")
        # Display the stitched image
        cv2.imshow("Stitched Image", stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    # Error handling
    print(f"Stitching failed. Error code: {error}")

    if error == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Error: Not enough images for stitching.")
    elif error == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Error: Homography estimation failed. Try using better or overlapping images.")
    elif error == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Error: Camera parameters adjustment failed.")
    else:
        print("Unknown error occurred during stitching.")

cv2.destroyAllWindows()
