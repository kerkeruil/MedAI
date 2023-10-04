import nibabel as nib
from pathlib import Path

data_folder = Path(r"C:\Users\aries\OneDrive - Vrije Universiteit Amsterdam\Msc AI\5204AFMI6Y - AI for Medical Imaging\images")
n1_path = data_folder / "RibFrac421-image.nii"
print(n1_path.read_text)

# n1_img = nib.load(r"C:/Users/aries/OneDrive - Vrije Universiteit Amsterdam/Msc AI/5204AFMI6Y - AI for Medical Imaging/images/RibFrac421-image.nii")
# n1_img = nib.load("C:/Users/aries/Downloads/T2-interleaved.nii/05aug14_test_samples_8_1.nii")
n1_img = nib.load(n1_path)

print(n1_img.header)

print(n1_img.affine)