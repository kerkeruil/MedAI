from vedo import load, show, Volume

from vedo import dataurl, Volume
from vedo.applications import IsosurfaceBrowser, RayCastPlotter


path_nifti = "./data/volume.nii.gz"
path_stl = "./data/output_file.stl"

image_path = "DATA_FULL/ribfrac-val-images/RibFrac421-image.nii.gz"

def visualize_nifti(path_to_file, bg=(1,1,1), mesh_color=(1,0,0)):
    # Load a NIfTI file
    vol = Volume(path_to_file)
    # Original
    # show(vol, bg=bg)


    # IsosurfaceBrowser(Plotter) instance:
    plt = IsosurfaceBrowser(vol, use_gpu=False, c='gold')
    plt.show(axes=7, bg2='lb').close()

    # Potential
    # plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)  # Plotter instance
    # plt.show(viewup="z").close()

# -200 --> 1000
# visualize_stl(path_stl)
visualize_nifti(image_path)