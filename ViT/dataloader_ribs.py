import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

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

    

if __name__ == "__main__":
    path_to_image = find_path_to_folder('dataset')
    # d = readin_slices(path_to_image, ['422'])
    d = readin_slices(path_to_image, ['422', '423'])
    slices = d['422']['frac_0']['neg'][:,:,:3]
    show_slices(slices)

