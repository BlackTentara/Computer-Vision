import numpy as np
import cv2 as cv

def generate_alpha_mask(image_path):
    image = cv.imread(image_path)
    mask = np.zeros(image.shape[:2], np.uint8)

    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    alpha = mask2 * 255

    return alpha

def cut_and_paste(image_path, background_path):
    image = cv.imread(image_path)
    alpha = generate_alpha_mask(image_path)
    foreground = cv.bitwise_and(image, image, mask=alpha)

    background = cv.imread(background_path)
    background = cv.resize(background, (image.shape[1], image.shape[0]))

    alpha_3c = cv.merge([alpha, alpha, alpha]) / 255.0
    composite = cv.convertScaleAbs(foreground * alpha_3c + background * (1 - alpha_3c))
    
    return composite

# Example usage
image_path = 'bird2.jpeg'
background_path = 'forest.jpg'
composite_image = cut_and_paste(image_path, background_path)

cv.imwrite('output_composite.png', composite_image)
