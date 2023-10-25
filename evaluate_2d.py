import torch
import matplotlib
import argparse
import importlib
import torchmetrics
from torch.utils.data import DataLoader, random_split
import os
from models.unet import UNet
from vitseg_grayscale3d import *
from collections import defaultdict
import sys
sys.path.insert(0, 'Pytorch_UNet/utils/')
from data_loading import BasicDataset

# Some global file variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

def dice_coefficient(pred, label):
    intersection = torch.sum(label * pred)
    return (2.0 * intersection) / (torch.sum(label) + torch.sum(pred))

def iou_score(pred, label):
    iou = torchmetrics.classification.BinaryJaccardIndex().to(device)

    return iou(pred, label)

def calc_f1(pred, label):
    prec = torchmetrics.Precision(task="binary", average="micro", num_classes=2).to(device)
    recall = torchmetrics.Recall(task="binary", average="micro", num_classes=2).to(device)
    f1 = torchmetrics.classification.BinaryF1Score().to(device)

    return prec(pred,label), recall(pred,label), f1(pred,label)


def evaluate_batch(pred, label):
    """
    Perform evaluation for a single batch.
    """
    dice = dice_coefficient(pred,label)

    iou = iou_score(pred, label)

    f1 = calc_f1(pred, label)

    return [dice, f1, iou]

def convert_slices_to_tensor(slices, labels):
    """
    Convert slice matrix output into tensors. Allows for consistent comparison
    between vitseg and UNet.
    
    params:
    - slices: list of slices created by create_slice_matrix()
    - labels: list of associated masks created by create_slice_matrix()
    outputs:
    - dataloader: list of tuples (imgs,labels) made from slices and labels.
    """

    dataloader = []
    for frac in range(len(slices)):
        img_t = np.zeros((4,1,64,64))
        mask_t = np.zeros((4,64,64))
        for slice in range(len(labels[frac])):
            # Load best slices
            img = slices[frac][slice]
            mask = labels[frac][slice]
            # Add dim
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            # Stack dims
            img_t[slice,:,:,:] = img
            mask_t[slice,:,:] = mask
        dataloader.append((torch.Tensor(img_t), torch.Tensor(mask_t)))

    return dataloader

def load_unet(data_dir, model_path, batch_size=16, dropout=False, load_best=True):
    """
    Initialize and load unet architecture & dataloader.
    
    params:
    - dir_img: path to preproc images
    - dir_mask: path to preproc masks
    - model-dir: str - path to model.pth file
    - batch-size: int - batch size
    - dropout: boolean - use dropout or not

    returns: 
    - model: pretrained unet object
    - dataloader: dataloader to evaluate model
    """
    
    # Init img and mask path
    dir_img = os.path.join(data_dir, "images")
    dir_mask = os.path.join(data_dir, "labels")

    # Init and load model
    model = UNet(n_channels=1, n_classes=2, bilinear = False, dropout=dropout)
    state_dict = torch.load(model_path, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)

    # Init dataset
    if not load_best:
        dataset = BasicDataset(dir_img, dir_mask, 0.5, mask_suffix="_mask")
        loader_args = dict(batch_size=batch_size, num_workers= os.cpu_count(), pin_memory=True)
        dataloader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    else:
        # dataloader = []
        total_best_slices, total_best_labels = create_slice_matrix(Path(data_dir))
        dataloader = convert_slices_to_tensor(total_best_slices, total_best_labels)


    return model, dataloader
    
def load_vitseg(data_dir, model_path):
    """
    Initialize and load vitseg architecture & dataloader.
    """

    # Init and load model
    model = torch.jit.load(model_path)

    # Init dataset
    total_best_slices, total_best_labels = create_slice_matrix(Path(data_dir))

    dataloader = make_vitseg_compatible(total_best_slices, total_best_labels)

    return model, dataloader

def print_results(eval_dict):

    print("EVALUATION SCORES")
    print("=" * 20)
    print("average dice:", sum(eval_dict['dice']) / len(eval_dict['dice']))
    print("-" * 20)
    prec = [i[0] for i in eval_dict['f1']]
    recall = [i[1] for i in eval_dict['f1']]
    f1 = [i[2] for i in eval_dict['f1']]
    print("average precision", sum(prec)/len(prec))
    print("average recall", sum(recall)/len(recall))
    print("average f1", sum(f1)/len(f1))
    print("-" * 20)
    print("average iou:", sum(eval_dict['iou'])/len(eval_dict['iou']))
    tp = [i[0] for i in eval_dict['roc']]
    fp = [i[1] for i in eval_dict['roc']]

    try:
        print("average tp rate:", sum(tp) / len(tp))
    except:
        print("average tp rate:", 0)
    
    try:
        print("average fp rate:", sum(fp) / len(fp))
    except:
        print("average fp rate:", 0)
    print("=" * 20)


def evaluate_model(model_name, data_dir, load_best=False):
    """
    Main evaluation function.
    Evaluate a model's performance by measuring the following score:
    - F1 score
    - IOU
    - DICE

    params:
    --model_name: name of model to use. can be 'unet' or 'vitseg'
    --data_dir: path to directory where "images/" and "labels/" are stored
    --load_best: boolean to indicate whether testing is done on the full test
    or if a label-dense subsection is used.

    returns:
    - eval_dict: dictionary with 'f1','iou', 'roc' and 'dice' as keys.
    """

    eval_dict = defaultdict(list)

    # Setting up model
    if model_name == 'unet':
        model, dataset = load_unet(data_dir, "checkpoints/vanilla_epoch100.pth", load_best=load_best)
    elif model_name == 'vitseg':
        model, dataset = load_vitseg(data_dir, "checkpoints/vitseg_50.pth", load_best=load_best)
    

    model.eval()
    model = model.to(device)
    
    print("Starting evaluation...")
    print("Using", device)

    # Evaluate model
    for batch in dataset:
        if model_name == 'unet' and not load_best:
            images, gt_masks = batch['image'], batch['mask']

        elif model_name == 'unet' and load_best:
            images, gt_masks = batch[0], batch[1]

        elif model_name == "vitseg" or model_name == "unet":
            images = dataset[0]
            gt_masks = dataset[1]

        images = images.to(device)
        gt_masks = gt_masks.to(device)

        preds = model(images)
        preds = torch.argmax(preds, dim=1)

        dice, f1, iou = evaluate_batch(preds, gt_masks)
        eval_dict['dice'].append(dice)
        eval_dict['f1'].append(f1)
        eval_dict['iou'].append(iou)
    return eval_dict


def get_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--model_name", type=str)
     parser.add_argument("--data_dir", type=str)
     return parser.parse_args()
     
     

if __name__ == '__main__':
    args = get_args()

    eval_dict = evaluate_model(args.model_name,
                               args.data_dir,
                               load_best=True)
    print_results(eval_dict)




        


    

