
# 1. Import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
import math
import numpy as np

# 3. Load the two input images
imageA = cv2.imread('D:\\Machine Learning and Pattern Recognition\\SRGAN-PyTorch-main\\lenna.png')
imageB = cv2.imread('D:\\Machine Learning and Pattern Recognition\\SRGAN-PyTorch-main\\ienna_4k.png')

# 4. Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 5. Compute the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# 6. You can print only the score if you want
print("SSIM: {}".format(score))

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(imageA, imageB)
print('PSNR: ' + str(d))

print('MSE: '+ str(np.mean((imageA - imageB) ** 2)))
