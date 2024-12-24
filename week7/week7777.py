import numpy as np
import cv2
import glob
import imutils

image_paths = glob.glob('week7/imagesNew/*.jpg')
images =[]

for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
    cv2.waitKey(0)

imageStitcher = cv2.Stitcher_create()

error,  stitched_img = imageStitcher.stitch(images)

if not error:
    cv2.imwrite("StitchedOutput.png", stitched_img)
    cv2.imshow("Stitched Image", stitched_img)
    cv2.waitKey(0)