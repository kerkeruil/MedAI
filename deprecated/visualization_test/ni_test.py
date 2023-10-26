import numpy as np
import nibabel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import sys
import os
image_path = "DATA_FULL/ribfrac-val-images/"
label_path = "DATA_FULL/ribfrac-val-labels/"
image_data_path = os.path.join(image_path, "RibFrac421-image.nii.gz")
label_data_path = os.path.join(label_path, "RibFrac421-label.nii.gz")


data = nibabel.load(image_data_path).get_fdata().astype(np.uint8)
# data = nibabel.load(image_data_path).get_fdata()
data2 = nibabel.load(label_data_path).get_fdata()

# print(data.shape)
# print(data.shape[2])
# print(np.zeros(data.shape).shape)
def prep_data(data):
    template = np.zeros(data.shape)
    for i in range(data.shape[2]):
        template[:,:,i] = cv2.equalizeHist(data[:,:,i])
    return template

scaled_data = prep_data(data)
print(np.max(scaled_data))
# img = nibabel.viewers.OrthoSlicer3D(scaled_data)
# img.show()


# img = data[:,:,59]
# print(type(img))
# print(type(img[0]))
# print(type(img[0][0]))
# print(img.shape)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)