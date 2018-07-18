import matplotlib.pyplot as plt

from matplotlib.image import imread

img = imread('c:/python/test.jpg')

plt.imshow(img)
plt.show()