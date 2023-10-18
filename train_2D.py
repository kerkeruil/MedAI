"""
File for training the 2D UNet model. UNet implementation by https://github.com/milesial/Pytorch-UNet.
"""

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_img", type=str, help="path to image directory")
    parser.add_argument("--dir_mask", type=str, help="path to label mask directory")
    parser.add_argument("--epochs", type=int, help="amount of epochs to train")
    parser.add_argument("--batch_size", type=int, dest="batch_size")
    parser.add_argument("--learning_rate", type=float, dest="lr")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.system("python Pytorch-UNet/train.py \
              --dir_img {}\
              --dir_mask {}\
              --epochs {}\
              --batch-size {}\
              --learning-rate {}\
              ".format(args.dir_img,
                       args.dir_mask,
                       args.epochs, 
                       args.batch_size,
                       args.lr))