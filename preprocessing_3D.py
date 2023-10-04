"""
Usage: python3 preprocessing_2D.py --gt_dir "path to ground truth files" --label_dir "path to ground truth labels"
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm
import skimage
import random

from challenge_code.nii_dataset import NiiDataset

def preprocess(gt_dir, label_dir):
    data = NiiDataset(gt_dir)
    labels = NiiDataset(label_dir)
    if not os.path.exists("dataset_3D"):
        os.makedirs("dataset_3D")

    for image in range(len(data)):
        test_data = data[image][0]

        # e.x. ribfrac421
        name = data[image][1]
        print("Collecting data from:", name)
        path_name = "dataset_3D/" + name
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        label_img = labels[image][0]
        fractures = skimage.measure.regionprops(label_img.astype('int'))


        for i, fracs in enumerate(fractures):
            # print(fracs.label)
            # print(fracs.coords)
            # print("centroid",fracs.centroid)
            centroid = np.int64(np.round(fracs.centroid))
            # print("centroid round",centroid)
            path = path_name + "/frac_" + str(i)

            # name_frac = name + "-" + str(i)
            # path = "dataset/" + str(name_frac)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            #     os.makedirs(path +"/pos")
            #     os.makedirs(path +"/neg")

            # Create the patch edges, of size 96x96 because of centroid we want to add 48 to each side of the centroid
            mid_difference = np.int64(96 / 2)
            x_max, y_max, z_max = np.max(fracs.coords, axis=0)
            x_min, y_min, z_min = np.min(fracs.coords, axis=0)
            x_start = centroid[0] - mid_difference
            x_end = centroid[0] + mid_difference
            y_start = centroid[1] - mid_difference
            y_end = centroid[1] + mid_difference
            z_start = centroid[2] - mid_difference
            z_end = centroid[2] + mid_difference


            full_fracture = test_data[x_start:x_end, y_start:y_end, z_start:z_end]
            full_fracture_label = label_img[x_start:x_end, y_start:y_end, z_start:z_end]
            x_rand = random.randint(0,32)
            y_rand = random.randint(0,32)
            z_rand = random.randint(0,32)

            pos_random_patch = full_fracture[x_rand: x_rand+64, y_rand: y_rand+64, z_rand: z_rand+64]
            pos_random_patch_label = full_fracture_label[x_rand: x_rand+64, y_rand: y_rand+64, z_rand: z_rand+64]
            np.save(path + "_pos",pos_random_patch)
            np.save(path + "_pos-label",pos_random_patch_label)

            # negative fracture
            neg_x_start = np.shape(test_data)[0] - x_start-96
            neg_x_end = neg_x_start + 96

            # neg_x_end = np.shape(test_data)[0] -x_end+96
            negative_sample = test_data[neg_x_start:neg_x_end, y_start:y_end, z_start:z_end] #96x96
            negative_sample_label = label_img[neg_x_start:neg_x_end, y_start:y_end, z_start:z_end] #96x96
            neg_mirror_start = np.shape(negative_sample)[0] - 64 - x_rand
            negative_sample_patch = negative_sample[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64, z_rand:z_rand+64]
            negative_sample_patch_label = negative_sample_label[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64,z_rand:z_rand+64]

            if not(np.mean(negative_sample_patch_label) > 0.0):
                np.save(path + "_neg",negative_sample_patch)
                # np.save(path_neg + "neg-slice-" + str(slice)+"-label",negative_sample_label)
            else:
                print(negative_sample)

    # slice_0 = test_data[260, :, :]
    # slice_1 = test_data[:, 200, :]
    # slice_2 = test_data[:, :, 160]
    # patches = create_patches(test_data)


def create_patches(arr, overlap_bool=False):
    patch_size = (64, 64, 64)

    overlap = 0
    if overlap_bool == True:
        overlap = 12
    
    # Calculate the number of patches along each dimension
    num_patches_x = arr.shape[0] // (patch_size[0] - overlap)
    num_patches_y = arr.shape[1] // (patch_size[1] - overlap)
    num_patches_z = arr.shape[2] // (patch_size[2] - overlap)
    patches = []
    # Extract patches using a loop
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            for k in range(num_patches_z):
                x_start = i * (patch_size[0] - overlap)
                x_end = x_start +  patch_size[0]
                y_start = j * (patch_size[1] - overlap)
                y_end = y_start + patch_size[1]
                z_start = k * (patch_size[2] - overlap)
                z_end = z_start + patch_size[2]

                patch = arr[x_start:x_end, y_start:y_end, z_start:z_end]
                patches.append(patch)

    # Convert the list of patches into a NumPy array
    patches_array = np.array(patches)
    return patches_array

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
       
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--label_dir", required=True)

    args = parser.parse_args()

    preprocess(args.gt_dir, args.label_dir)
