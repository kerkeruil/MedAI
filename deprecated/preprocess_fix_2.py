import os, shutil
import numpy as np
import skimage
import random
import sys
from tqdm import tqdm

from PIL import Image


from challenge_code.nii_dataset import NiiDataset
from ViT.dataloader_ribs import find_path_to_folder
from sklearn.model_selection import train_test_split

import matplotlib

def patch_imsize(im):
    canvas = np.zeros((64,64))
    # Define the start location of paste operation
    target_row = 0  
    target_col = 0

    # Get shape of image
    im_rows, im_cols = im.shape

    # Paste im into the canvas
    canvas[target_row:target_row + im_rows, target_col:target_col + im_cols] = im
    
    return canvas 

def write_to_file(data, name):
    # for i, im in enumerate(data):
    filename = name + '.jpeg'
    # img = Image.fromarray(data.astype('uint8'), 'L')
    # img.save(filename)
    matplotlib.image.imsave(filename, data)

def preprocess(data_folder, destination_folder, partition, split_to_test=None):
    gt_dir = find_path_to_folder(os.path.join(data_folder, partition, 'images'))
    label_dir = find_path_to_folder(os.path.join(data_folder, partition, 'labels'))

    data = NiiDataset(gt_dir)
    labels = NiiDataset(label_dir)

    if not os.path.exists(destination_folder):  
        # Create correct folders
        folders = ['train', 'valid', 'test']
        for name in folders:
            os.makedirs(os.path.join(destination_folder, name, 'pos'))
            os.makedirs(os.path.join(destination_folder, name, 'neg'))


    for image in tqdm(range(len(data)), desc=f"Processing images from {partition} folder"):
        image_data = data[image][0]
        label_img = labels[image][0]
        im_ind = data[image][1][7:10]
        
        fractures = skimage.measure.regionprops(label_img.astype('int'))
        
        if not split_to_test:
            for i, fracs in enumerate(fractures):
                pos_patch_list, pos_label_list, neg_patch_list = create_fractures_slice(fracs, image_data, label_img, im_ind)
                name = im_ind + '_' + str(i)
                write_to_file(np.concatenate(pos_patch_list, axis=1), os.path.join(destination_folder, partition, 'pos', name))
                write_to_file(np.concatenate(neg_patch_list, axis=1), os.path.join(destination_folder, partition, 'neg', name))
        else:
            threshold = len(fractures)//2 - 1
            for i, fracs in enumerate(fractures):
                pos_patch_list, pos_label_list, neg_patch_list = create_fractures_slice(fracs, image_data, label_img, im_ind)
                name = im_ind + '_' + str(i)
                if i > threshold:
                    partition = 'test'
                write_to_file(np.concatenate(pos_patch_list, axis=1), os.path.join(destination_folder, partition, 'pos', name))
                write_to_file(np.concatenate(neg_patch_list, axis=1), os.path.join(destination_folder, partition, 'neg', name))



def create_fractures_slice(fracs, image, label_img, im_ind):
    pos_patch_list, pos_label_list, neg_patch_list = [], [], []
    centroid = np.int64(np.round(fracs.centroid))

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
    for slice, j in enumerate(range(z_min, z_max)):
        x_rand = random.randint(0,32)
        y_rand = random.randint(0,32)
        patch = image[x_start:x_end, y_start:y_end, j]
        patch_label = label_img[x_start:x_end, y_start:y_end,j]
        pos_random_patch = patch[x_rand: x_rand+64, y_rand: y_rand+64]
        pos_random_patch_label = patch_label[x_rand: x_rand+64, y_rand: y_rand+64]
        pos_random_patch_label[pos_random_patch_label > 0] = 1

        # Ensure all images have the correct shape
        if pos_random_patch.shape != (64,64):
            # print("Image:", im_ind)
            # print("Before:", pos_random_patch.shape)
            pos_random_patch = patch_imsize(pos_random_patch)
            # print("After", pos_random_patch.shape)
        pos_patch_list.append(pos_random_patch)
        pos_label_list.append(pos_random_patch_label)

        # negative slices
        neg_x_start = np.shape(image)[0] - x_start-96
        neg_x_end = np.shape(image)[0] -x_end+96
        negative_sample = image[neg_x_start:neg_x_end, y_start:y_end, j]
        negative_sample_label = label_img[neg_x_start:neg_x_end, y_start:y_end, j]
        neg_mirror_start = np.shape(negative_sample)[0] - 64 - x_rand
        negative_sample_patch = negative_sample[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64]
        negative_sample_patch_label = negative_sample_label[neg_mirror_start: neg_mirror_start+64, y_rand: y_rand+64]

        if negative_sample_patch.shape != (64,64):
            # print("Image:", im_ind)
            # print("Before:", negative_sample_patch.shape)
            negative_sample_patch = patch_imsize(negative_sample_patch)
            # print("After", negative_sample_patch.shape)

        neg_patch_list.append(negative_sample_patch)

    return pos_patch_list, pos_label_list, neg_patch_list


if __name__ == "__main__":
    # Ensure this works on all operating systems
    data_folder = 'raw_data'
    destination_folder = 'dataset_huggingface_2'

    # Remove old data
    if os.path.exists(destination_folder):  
        shutil.rmtree(destination_folder)

    preprocess(data_folder, destination_folder, 'train')
    preprocess(data_folder, destination_folder, 'valid', split_to_test=0.5)

    # preprocess(data_folder, destination_folder, 'valid')