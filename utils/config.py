import os
import torch
import random
import numpy as np
import setproctitle
from argparse import ArgumentParser
from torch import cuda, device as Device

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
setproctitle.setproctitle("NGPnet")

parser = ArgumentParser(description="Lung Nodule Growth Prediction")
parser.add_argument("--pred_model", type=str, default="NGPnet",
                    help="name of the segmentation model")
parser.add_argument("--device", type=Device, help="runtime device",
                    default=Device("cuda" if cuda.is_available() else "cpu"))

############################ save path ############################
parser.add_argument("--data_root", type=str, default="../datasets/Hospital-Seg",
                    help="root of the dataset")
parser.add_argument("--json_name", type=str, default="NGP3T.json",
                    help="name of json file in cross validation")
parser.add_argument("--num_folds", type=int, default=5,
                    help="number of folds in cross validation")
parser.add_argument("--fold", type=int, default=0,
                    help="fold of cross validation")

parser.add_argument("--model_save_dir", type=str, default="pretrained",
                    help="save path of trained models")
parser.add_argument("--trained_model", type=str, default="best_model.pth",
                    help="filename of pretrained model")
parser.add_argument("--log_save_dir", type=str, default="logs",
                    help="save path of runtime logs")
parser.add_argument("--pred_save_dir", type=str, default="prediction",
                    help="filename of prediction results")

############################ training ############################
parser.add_argument("--batch_size", type=int, default=4,
                    help="batch size of training")
parser.add_argument("--max_epochs", type=int, default=200,
                    help="max number of training epochs")
parser.add_argument("--val_freq", type=int, default=2,
                    help="validation frequency")
parser.add_argument("--num_workers", type=int, default=8,
                    help="number of workers")

parser.add_argument("--optim_name", type=str, default="AdamW",
                    help="name of optimizer")
parser.add_argument("--optim_lr", type=float, default=1e-3,
                    help="learning rate of optimizer")
parser.add_argument("--reg_weight", type=float, default=1e-4,
                    help="regularization weight")
parser.add_argument("--lrschedule", type=str, default="WarmUpCosine",
                    help="name of learning rate scheduler")
parser.add_argument("--warmup_steps", type=int, default=10,
                    help="number of learning rate warmup steps")

######################### preprocessing #########################
parser.add_argument("--s_min", type=float, default=-1200,
                    help="min value of original input images")
parser.add_argument("--s_max", type=float, default=600,
                    help="max value of original input images")
parser.add_argument("--r_min", type=float, default=0,
                    help="min value of normalized input images")
parser.add_argument("--r_max", type=float, default=1,
                    help="max value of normalized input images")

######################### network #########################
parser.add_argument("--in_size", type=int, default=64,
                    help="spatial size of input images")
parser.add_argument("--feat_dim", type=int, default=8,
                    help="channels of basic feature map")
parser.add_argument("--depths", type=tuple, default=(2,2,2,2),
                    help="dimension of time embedding")

######################### loss & metric #########################
parser.add_argument("--w_out", type=float, default=0.6,
                    help="weight coef for outside points in masked loss")
parser.add_argument("--expand", type=float, default=1.2,
                    help="expansion coef for ROIs in masked loss")

parser.add_argument("--w0_sim", type=float, default=2.0,
                    help="weight of similarity loss")
parser.add_argument("--w1_seg", type=float, default=1.0,
                    help="weight of segmentation loss")
parser.add_argument("--w2_reg", type=float, default=2.0,
                    help="weight of regularization loss")
parser.add_argument("--w3_smooth", type=float, default=1.0,
                    help="weight of smoothness loss")

SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

args = parser.parse_args()