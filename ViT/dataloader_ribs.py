import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from datasets import load_dataset

import matplotlib.image

from sklearn.model_selection import train_test_split

from pathlib import Path
import os

def show_slices(slices):
    """ Function to display row of image slices """
    n_slices = slices.shape[2]
    fig, axes = plt.subplots(1, n_slices)
    for i in range(n_slices):
        axes[i].imshow(slices[:,:,i], cmap="gray", origin="lower")
    plt.show()


def create_slice_matrix(path_to_image, fracture, label: str):
    """
    Read in all images of a given folder. Returns a matrix with all the images 
    stacked in the 3e dimension.
    """
    im_size = 64
    path_to_slices = os.path.join(path_to_image, fracture, label)
    slices = natsorted(os.listdir(path_to_slices))
    tmp_matrix = np.zeros((im_size, im_size, len(slices)))

    for i, s in enumerate(slices):
        pts = os.path.join(path_to_slices, s) # pts is path_to_slice  
        tmp_matrix[:, :, i] = np.load(pts)

    return tmp_matrix


def readin_slices(path_to_image_folder: Path, image_inds: list = None) -> dict:
    """
    Creates and returns a dictionary with 3d numpy arrays of stacked images.
    Works as follows:

    dict["Index of image"]["Index of fracture]["neg/pos_image/pos_label"]
    
    Where the 3e dim (e.g: im[:,:,x]) is equal to the index of the slices.
    """
    # Locate image folder containing the fractures.
    all_paths = os.listdir(path_to_image_folder)
    filenames = [(p,ind) for p in all_paths for ind in image_inds if ind in p]

    d = {}
    for name, ind in filenames:
        d[ind] = {}
        path_to_image = path_to_image_folder.joinpath(name)

        # Iterate over fractures.
        fracs = os.listdir(path_to_image)
        for f in fracs:
            neg_matrix = create_slice_matrix(path_to_image, f, 'neg')
            pos_image_matrix = create_slice_matrix(path_to_image, f, 'pos_image')
            pos_label_matrix = create_slice_matrix(path_to_image, f, 'pos_label')

            d[ind][f] = {'neg': neg_matrix, 'pos_image': pos_image_matrix, 'pos_label': pos_label_matrix}
    
    return d


def find_path_to_folder(tag):
    """
    Find folder that contains given tag.
    Returns the local path to this folder.
    """
    tag = str(Path(tag))
    workdir = os.getcwd()
    print(f'\nLooking for {tag} in: {workdir}')
    n = len(tag)
    found = False
    for (dir_path, dir_names, file_names) in os.walk(workdir , topdown=True):
      if tag == dir_path[-n:]:
        print(f'Found {dir_path}\n')
        found = dir_path
        break

    if not found:
        raise Exception("Couldn't find the folder")
    return Path(found)


def write_to_file(data, name):
    for i, im in enumerate(data):
        filename = name + str(i) + '.jpeg'
        matplotlib.image.imsave(filename, im)
    
def make_dataset_huggingface_compatible(d:dict):
    """
    Has to be run from MEDAI home folder.
    """
    test_ratio = .2 # Automatically calculate other sets. (train = 1-test, val = 0.5 of test)
    folders = ['train/', 'valid/', 'test/']
    # Ensure folders are in place
    folder_name = 'dataset_huggingface/'
    if not os.path.exists(folder_name):
        for name in folders:
            os.makedirs(folder_name + name + 'neg/')
            os.makedirs(folder_name + name + 'pos/')

    pos_examples = []
    neg_examples = []
    for im in d.keys():
        for frac in d[im].keys():
            # Store all slices in same folder. Separate for neg and pos due to different lengths
            for slice in range(d[im][frac]['pos_image'].shape[2]):
                pos_examples.append(d[im][frac]['pos_image'][:,:,slice])

            for slice in range(d[im][frac]['neg'].shape[2]):
                neg_examples.append(d[im][frac]['neg'][:,:,slice])
    
    # Split into sets
    for example, label in zip([pos_examples, neg_examples], ['pos/', 'neg/']):
        train_data, test_data = train_test_split(example, test_size=test_ratio, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

        # Add to folders
        for data, f in zip([train_data, val_data, test_data], folders):
            write_to_file(data, folder_name + f + label)

def create_dataset():
    dataset_path = Path("dataset_huggingface/")
    ds = load_dataset("imagefolder", drop_labels=False, keep_in_memory=True, data_dir=str(dataset_path))
    return ds

if __name__ == "__main__":
    path_to_image = find_path_to_folder('dataset')
    d = readin_slices(path_to_image, ['422'])
    # d = readin_slices(path_to_image, ['422', '423'])
    make_dataset_huggingface_compatible(d)
    # slices = d['422']['frac_0']['neg'][:,:,:3]
    # show_slices(slices)

