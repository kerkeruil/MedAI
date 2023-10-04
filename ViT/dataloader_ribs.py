import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
import os
import pathlib

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

def readin_slices(path_to_image: str, image_inds: list = None) -> dict:
    all_paths = os.listdir(path_to_image)
    paths = [p for p in all_paths for ind in image_inds if ind in p]
    
    for p in paths:
        pass



    # if image_ind:
    #     dir_name = [p for p in all_paths if image_ind in p]
    #     if len(dir_name) !=1:
    #         raise Exception(f"Found multiple versions of this slice.\n {dir_name}")
    #     else:
    #         dir_name = all_paths

    # for d in dir_name:


        
        

def find_path_to_folder(tag):
    """
    Find folder that contains given tag.
    Returns the local path to this folder.
    """
    workdir = os.getcwd()
    print(f'Looking for {tag} in: {workdir}')
    n = len(tag)
    found = False
    for (dir_path, dir_names, file_names) in os.walk(workdir , topdown=True):
    #   print(dir_path[-n:])
      if tag == dir_path[-n:]:
        print(f'Found {dir_path}\n')
        found = dir_path
        break

    if not found:
        raise Exception("Couldn't find the folder")
    return found


    

if __name__ == "__main__":
    path_to_image = find_path_to_folder('dataset')
    readin_slices(path_to_image, ['422', '423'])

    # im = [np.load('ViT/test.npy')]
    # im.append(np.load('ViT/test2.npy'))
    # show_slices(im)

