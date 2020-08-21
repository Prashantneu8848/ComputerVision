import cv2
from matplotlib import pyplot as plt
import numpy as np

"""
Grayscale image contains only rows and cols.
cv2 works with BGR while matplotlib works with RGB.
"""
image = cv2.imread("background.jpg")

# Get python type and datatype for the image.
print(type(image))
print(image.dtype)

#Get the dimension of the image.
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))


#Draw a diagonal line of 9 px in the image.
image = cv2.line(image, (0, 0), (w, h), (0, 255, 0), 9)

# Only get the green channel.
image = image[:,:,1]

#Cropping the grayscale image
cropped = image[10:20, 100:200]

#Adding a Gaussian blur filter to image.
hsize = (31, 31) # kernel size is 31 * 31
sigma = 5
blurimg = cv2.GaussianBlur(image, hsize, sigma)

imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imgrgb)
plt.show()
