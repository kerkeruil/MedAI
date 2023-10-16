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
from ViT.dataloader_ribs import find_path_to_folder
def preprocess():
    gt_dir = find_path_to_folder('MedAI_oefenpakket/images')
    label_dir = find_path_to_folder('MedAI_oefenpakket/labels')
    data = NiiDataset(gt_dir)
    labels = NiiDataset(label_dir)
    if not os.path.exists("dataset_test"):
        os.makedirs("dataset_test")

    # for image in range(len(data)):
    for image in range(2):
        test_data = data[image][0]

        # e.x. ribfrac421
        name = data[image][1]
        print("Collecting data from:", name)
        path_name = "dataset_test/images"
        path_name_label = "dataset_test/labels"
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        if not os.path.exists(path_name_label):
            os.makedirs(path_name_label)
        label_img = labels[image][0]
        fractures = skimage.measure.regionprops(label_img.astype('int'))


        for j, fracs in enumerate(fractures):
            # print(fracs.label)
            # print(fracs.coords)
            # print("centroid",fracs.centroid)
            centroid = np.int64(np.round(fracs.centroid))
            # print("centroid round",centroid)
            # path = path_name + "/frac_" + str(i)

            # # name_frac = name + "-" + str(i)
            # # path = "dataset/" + str(name_frac)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            #     os.makedirs(path +"/pos_image")
            #     os.makedirs(path +"/pos_label")
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

            # Create positive and negative slices
            for slice, i in enumerate(range(z_min, z_max)):
                x_rand = random.randint(0,32)
                y_rand = random.randint(0,32)
                patch = test_data[x_start:x_end, y_start:y_end, i]
                patch_label = label_img[x_start:x_end, y_start:y_end,i]
                pos_random_patch = patch[x_rand: x_rand+64, y_rand: y_rand+64]
                pos_random_patch_label = patch_label[x_rand: x_rand+64, y_rand: y_rand+64]
                np.save(path_name + "/" + name + "-frac-" + str(j) + "-slice-" + str(slice)+ "-pos",pos_random_patch)
                np.save(path_name_label + "/" + name  + "-frac-" + str(j) + "-label-slice-label" + str(slice)+"-pos",pos_random_patch_label)

                # negative slices
                neg_x_start = np.shape(test_data)[0] - x_start-96
                neg_x_end = np.shape(test_data)[0] -x_end+96
                negative_sample = test_data[neg_x_start:neg_x_end, y_start:y_end, i]
                negative_sample_label = label_img[neg_x_start:neg_x_end, y_start:y_end, i]
                neg_mirror_start = np.shape(negative_sample)[0] - 64 - x_rand
                negative_sample_patch = negative_sample[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64]
                negative_sample_patch_label = negative_sample_label[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64]

                # Save the negative slices
                path_neg = path_name


                if not(np.mean(negative_sample_patch_label) > 0.0):
                    np.save(path_neg + "/" + name +"-frac-" + str(j) + "-slice-" + str(slice)+"-neg",negative_sample_patch)
                    np.save(path_name_label + "/" + name + "-frac-" + str(j) +"-label-slice-" + str(slice)+"-neg",negative_sample_patch)
                    # np.save(path_neg + "neg-slice-" + str(slice)+"-label",negative_sample_label)
                else:
                    print(negative_sample)
                # show_slices(np.array([random_patch,random_patch_label]))
    
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

if __name__ == "__main__":
    preprocess()
