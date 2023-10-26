import numpy as np

# Set seed for consistency in checks.
np.random.seed(42)

rnd_matrix = np.random.randint(0, high=10, size=(3,4, 2), dtype='l')


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dikke_koe.jpg',0)
print(type(img))
print(type(img[0]))
print(type(img[0][0]))
print(img.shape)
equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)tho