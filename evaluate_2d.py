import torch
import matplotlib
import argparse
import importlib
from sklearn.metrics import jaccard_score, f1_score
from sklearn import metrics
from torch.utils.data import DataLoader, random_split
import os
from models.unet import UNet
from collections import defaultdict

import sys
sys.path.insert(0, 'Pytorch_UNet/utils/')
from data_loading import BasicDataset

# from MedAI.Pytorch_UNet.utils.data_loading import BasicDataset



def dice_coefficient(pred, label):
    intersection = torch.sum(label * pred)
    return (2.0 * intersection) / (torch.sum(label) + torch.sum(pred))

def iou_score(pred, label):
    return jaccard_score(pred, label, average='micro')

def calc_f1(pred, label):
    return f1_score(label, pred, average='micro')

def roc_score(pred, label):
    # fpr, tpr, thresholds = metrics.roc_curve(pred, label, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(pred, label, pos_label=1)

    return fpr, tpr

def evaluate_batch(pred, label):
    """
    Perform evaluation for a single batch.
    """
    dice = dice_coefficient(pred,label)

    iou = iou_score(pred, label)

    f1 = calc_f1(pred, label)

    roc = roc_score(pred,label)

    return [dice, f1, iou, roc]

def evaluate_model(model_path, batch_size, dir_img, dir_mask):
    """
    Evaluate a model's performance by measuring the following score:
    - F1 score
    - IOU
    - ROC
    - DICE

    params:
    --model_path: path to model to train
    --dir_img: path to directory where images are stored
    --dir_mas: path to directory where label masks are stored

    returns:
    - eval_dict: dictionary with 'f1','iou', 'roc' and 'dice' as keys.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_dict = defaultdict(list)

    # Setting up dataset
    dataset = BasicDataset(dir_img, dir_mask, 0.5, mask_suffix="_mask")
    n_val = int(len(dataset) * 0.2)
    n_test = n_val
    n_train = len(dataset) - n_val*2

    _, _, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size, num_workers= os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    # Setting up model
    model = UNet(n_channels=1, n_classes=2, bilinear = False, dropout=False)
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.eval()

    print("Starting evaluation...")
    # Evaluate model
    for batch in test_loader:
        images, gt_masks = batch['image'], batch['mask']

        images = images.to(device)
        gt_masks = gt_masks.to(device)

        preds = model(images)
        print(preds.shape, gt_masks.shape)

        dice, f1, iou, roc = evaluate_batch(preds, gt_masks)
        eval_dict['dice'].append(dice)
        eval_dict['f1'].append(f1)
        eval_dict['iou'].append(iou)
        eval_dict['roc'].append(roc)

    print("EVALUATION SCOREs")
    print("average dice:", sum(eval_dict['dice'])/len(eval['dice']))
    print("average f1:", sum(eval_dict['f1'])/len(eval['f1']))
    print("average iou:", sum(eval_dict['iou'])/len(eval['iou']))
    print("average roc:", sum(eval_dict['roc'])/len(eval['roc']))

    return eval_dict


def get_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--model_path", type=str)
     parser.add_argument("--batch_size", type=int)
     parser.add_argument("--dir_img", type=str)
     parser.add_argument("--dir_mask", type=str)

     return parser.parse_args()
     
     

if __name__ == '__main__':
    args = get_args()

    eval_dict = evaluate_model(args.model_path,
                               args.batch_size,
                               args.dir_img,
                               args.dir_mask
                               )




        


    

