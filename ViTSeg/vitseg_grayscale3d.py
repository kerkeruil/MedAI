import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from natsort import natsorted


import vitseg_dataprep as vitsegdata
import torchvision.transforms as transforms
from PIL import ImageOps


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def print_title(title):
    title_len = len(title)
    dashes = "".join(["-"] * title_len)
    print(f"\n{title}\n{dashes}")


# Model setup
# ----------------------------------------
class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        return y


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x


class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        """in_channels is the number of input channels in the input that will be
        fed into this layer. For RGB images, this value would be 3.
        """
        super().__init__()
        self.i2p = ImageToPatches(image_size, patch_size)
        self.pe = PatchEmbedding(patch_size * patch_size * in_channels, embed_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embed = nn.Parameter(torch.randn(num_patches, embed_size))

    def forward(self, x):
        x = self.i2p(x)
        x = self.pe(x)
        x = x + self.position_embed
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)


class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)
        self.fold = nn.Fold(
            output_size=(image_size, image_size),
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x


class VisionTransformerForSegmentation(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        out_channels,
        embed_size,
        num_blocks,
        num_heads,
        dropout,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        heads = [
            SelfAttentionEncoderBlock(embed_size, num_heads, dropout)
            for i in range(num_blocks)
        ]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


@dataclass
class VisionTransformerArgs:
    """Arguments to the VisionTransformerForSegmentation."""

    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3  # Number of layers in image (RGB=3)
    out_channels: int = (3)  # Ouput number of layers
    embed_size: int = 768
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.2


# Setup Training
# --------------------------------------
# Train the model for a single epoch
def train_model(model, loader, optimizer):
    to_device(model.train())

    criterion = nn.CrossEntropyLoss(reduction="mean")

    running_loss = 0.0
    running_samples = 0

    inps, targs = loader
    for batch_idx, (inputs, targets) in enumerate(zip(inps, targs), 0):
        optimizer.zero_grad()
        targets = targets.type(torch.long)

        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)

        targets = targets.squeeze(dim=1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()

    print(
        "Trained {} samples, Loss: {:.4f}".format(
            running_samples,
            running_loss / (batch_idx + 1),
        )
    )


def train_loop(model, loader, epochs, optimizer, scheduler, save_path):
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)

        if scheduler is not None:
            scheduler.step()
        print("")
    


def stack_and_stitch(tensors):
    """
    Make one tensor out of 4 individual tensors.
    """
    t1 = torch.from_numpy(tensors[0])
    t2 = torch.from_numpy(tensors[1])
    t3 = torch.from_numpy(tensors[2])
    t4 = torch.from_numpy(tensors[3])
   
    for t in [t1, t2, t3, t4]:
        if t.shape != (64,64):
            print(t.shape)
            print(t)

    horizontal_stack = torch.cat((t1, t2), dim=1) #  2x4 tensor
    horizontal_stack2 = torch.cat((t3, t4), dim=1) # 2x4 tensor
    return torch.cat((horizontal_stack, horizontal_stack2), dim=0) # Connect horizontal tensors

def make_vitseg_compatible(data_patches, data_labels):
    samples = len(data_patches)
    inputs_batch = torch.zeros((samples, 3, 128, 128))
    segmask_batch = torch.zeros((samples, 128, 128))
    for i, (ims, lbs) in enumerate(zip(data_patches, data_labels)):
        full_im = stack_and_stitch(ims)
        full_lab = stack_and_stitch(lbs)

        inputs_batch[i, :, :, :] = full_im
        segmask_batch[i, :, :] = full_lab

    return inputs_batch, segmask_batch

    
def create_slice_matrix(path_to_image):
    """
    Read in all images of a given folder. Returns a matrix with all the images 
    stacked in the 3e dimension.
    """
    im_size = 64
    path_to_slices = os.path.join(path_to_image, "images")
    path_to_labels = os.path.join(path_to_image, "labels")
    slices = natsorted(os.listdir(path_to_slices))
    name = slices[0][:10]
    total_best_slices = []
    total_best_labels = []
    best_slices = []
    best_slices_labels = [] 
    j = 0
    for i, s in enumerate(slices):
        # check if the ribcase changed
        if s[:23] != name[:23]:
            if len(best_slices) == 4:
                total_best_slices.append(best_slices)
                total_best_labels.append(best_slices_labels)
            best_slices = []
            best_slices_labels = []
            name = s
            j = 0
        pts = os.path.join(path_to_slices, s)
        single_slice = np.load(pts)
        label_file_name = s[:-4] + "_mask.npy"
        label_img = os.path.join(path_to_labels,label_file_name)
        single_slice_label = np.load(label_img)
        # put the first 4 in the list
        if j < 4:
            best_slices.append(single_slice)
            best_slices_labels.append(single_slice_label)
        else:
            no_of_pixels = np.sum(single_slice_label)
            # calculate the array with the least amount of fractures and it's index
            # print(len(best_slices_labels))
            min_array = min(best_slices_labels, key=lambda x: np.sum(x))
            min_index = min(range(len(best_slices_labels)), key=lambda k: np.sum(best_slices_labels[k]))
            if no_of_pixels > np.sum(min_array):
                best_slices_labels[min_index] = single_slice_label
                best_slices[min_index] = single_slice
        j += 1
    return total_best_slices[1:], total_best_labels[1:]

import sys


if __name__ == "__main__":
    total_best_slices, total_best_labels = create_slice_matrix(Path('dataset_manual_test/train'))
    
 
    # Create dummy images and labels
    # ----------------------------------------------------
    # data_patches = [] 
    # data_labels = []

    # num_ones = 20 

    # im_size = 64
    # samples = 3

    # min_val = 0
    # max_val = im_size * im_size
    # for i in range(samples):
    #     patches = []
    #     labels = []        
    #     for j in range(4):
    #         dummy_image = torch.arange(min_val, max_val).reshape((im_size, im_size))
    #         dummy_label = torch.zeros((im_size, im_size))
    #         min_val = max_val
    #         max_val = max_val + im_size*im_size

    #         # Randomly place ones
    #         indices = torch.randint(0, im_size * im_size, (num_ones,))
    #         dummy_label.view(-1)[indices] = 1

    #         patches.append(dummy_image.numpy()) 
    #         labels.append(dummy_label.numpy())

    #     data_patches.append(patches)
    #     data_labels.append(labels)
    
    # inputs_batch_old, segmask_batch_old = make_vitseg_compatible(data_patches, data_labels)

    
    # End dummy inputs and labels
    # ----------------------------------------------------


    print("Creating vitseg compatible data")
    print(f"Found {len(total_best_slices)} total_best_slices.")
    print(f"Found {len(total_best_labels)} total_best_labels.")

    inputs_batch, segmask_batch = make_vitseg_compatible(total_best_slices, total_best_labels)

    print("Succesfully created input data with shapes:")
    print(f"Input batch: {inputs_batch.shape}")
    print(f"Segmask batch: {segmask_batch.shape}")


    # Set up training params
    # -----------------------------------------
    vit_args = dataclasses.asdict(VisionTransformerArgs())
    vit = VisionTransformerForSegmentation(**vit_args)

    m = vit
    to_device(m)

    images_folder_name = "vit_training_progress_images"
    save_path = os.path.join("output", images_folder_name)
    optimizer = torch.optim.Adam(m.parameters(), lr=0.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)

    # Run model
    print("MODELTYPE: ", type(m))
    # sys.exit()
    print("Start training loop")
    train_loop(
        m,
        ([inputs_batch], [segmask_batch]),
        (1, 3),
        optimizer,
        scheduler,
        save_path=save_path,
    )

    model_scripted = torch.jit.script(m) # Export to TorchScript
    model_scripted.save('model_scripted.pt') # Save
