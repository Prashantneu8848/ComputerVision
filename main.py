import cv2
import math
from matplotlib import pyplot as plt
import numpy as np

def display_image(image, msg):
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(msg)
    plt.imshow(imgrgb)
    plt.show()


"""
Grayscale image contains only rows and cols.
cv2 works with BGR while matplotlib works with RGB.
"""
image = cv2.imread("background.jpg")
display_image(image, 'Original Image')

imgCopy2 = image.copy()

# Get python type and datatype for the image.
print(type(image))
print(image.dtype)

#Get the dimension of the image.
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))


# Draw a diagonal line of 9 px in the image.
# image = cv2.line(image, (0, 0), (w, h), (0, 255, 0), 9)

# Only get the green channel.
image = image[:,:,1]
display_image(image, 'image only with green channel')

# Cropping the grayscale image
cropped = image[10:20, 100:200]

# Adding a Gaussian blur filter to image.
hsize = (31, 31) # kernel size is 31 * 31
sigma = 5
blurimg = cv2.GaussianBlur(image, hsize, sigma)
display_image(blurimg, 'image with gaussian blur')

# Adding a median filter to image with kernel size 5.
medianBlur = cv2.medianBlur(image, 5)
display_image(medianBlur, 'image with median blur')

# Matching template in the image.
imgCopy = image.copy()
res = cv2.matchTemplate(imgCopy, cropped, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
bottom_right = (max_loc[0] + w, max_loc[1] + h)
cv2.rectangle(imgCopy, max_loc, bottom_right, 255, 2)
display_image(imgCopy, 'image matching template from cropped image')

'''
Find x and y gradient using Sobel Kernel.

The initial image is unit8. Black-to-White transition is taken as Positive slope
while White-to-Black transition is taken as a Negative slope. So when it is converted to uint8,
all negative slopes are made zero and the edge is lost. So, a higher form like 16s ot  64f is
chosen.
'''
sobel_x_gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobel_y_gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1)
mag, direction = cv2.cartToPolar(sobel_x_gradient, sobel_y_gradient, angleInDegrees=True)

# Edge detection using Canny Edge Detection Algorithm which has magnitude
# threshold of 100 min and 200 max.
canny_edge = cv2.Canny(image, 100, 200)

# Use Standard Hough Transform to find candidate for lines.
lines = cv2.HoughLines(canny_edge, 1, np.pi / 180, 150, None, 0, 0)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        # Add lines to the edges detected from Canny Edge.
        cv2.line(canny_edge, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

display_image(canny_edge, 'edge detection using canny edge algorithm')

# TODO (me) add a picture with circles and use Hough Transform for Circles.

# TODO (me) add stereo picture and show epipolar geometry between the pictures.

# TODO (me) use SIFT to find corners in image.


# use Harris Corner Detection Algorithm to find corners in image.
gray = cv2.cvtColor(imgCopy2, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
imgCopy2[dst>0.01*dst.max()]=[0,0,255]
# imgCopy2rgb = cv2.cvtColor(imgCopy2, cv2.COLOR_BGR2RGB)
display_image(imgCopy2, 'harris corner detection')
