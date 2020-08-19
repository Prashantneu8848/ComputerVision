import cv2
from matplotlib import pyplot as plt

image = cv2.imread("background.jpg")

#Get the dimension of the image.
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

#Draw a diagonal line of 9 px in the image.
image = cv2.line(image, (0, 0), (w, h), (0, 255, 0), 9)

# Only get the green channel.
image = image[:,:,1]


imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imgrgb)
plt.show()
