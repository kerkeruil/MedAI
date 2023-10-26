import os
import numpy as np

s = np.arange(10)
print(s)

data_path = "DATA_FULL/ribfrac-val-images/"
example_filename = os.path.join(data_path, "RibFrac421-image.nii.gz")

from nibabel.testing import data_path

import nibabel as nib
img = nib.load(example_filename).get_fdata()

a = img.shape
print(a)

import matplotlib.pyplot as plt


test = img[:,:,59]
plt.imshow(test, cmap="gray")
plt.show()


