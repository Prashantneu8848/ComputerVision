from cv2 import *
from matplotlib import pyplot as plt

image = cv2.imread("background.jpg")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imgrgb)
plt.show()
