import os
import time
import torch
import pandas as pd

from argparse import ArgumentParser
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from nn.losses import *
from nn.metrics import *
from nn.ngpnet import NGPnet


def get_pred_model(args: ArgumentParser, pretrained: bool = True):
    ''' get growth prediction model '''
    model_name = args.pred_model.lower()
    if model_name == "ngpnet":
        pred_model = NGPnet(img_size=args.in_size,
                            feat_dim=args.feat_dim,
                            depths=args.depths)
    else:
        raise ValueError(
            f"Growth Prediction Model `{args.pred_model}` is not supported!"
        )
    pred_model = pred_model.to(args.device)

    if pretrained:
        path = os.path.join(args.model_save_dir,
                            args.pred_model, args.trained_model)
        state_dict = torch.load(path)
        pred_model.load_state_dict(state_dict["net"])
        print("LOAD checkpoints from `%s`..." % path)
    return pred_model


def get_optim(model: Module, args: ArgumentParser):
    ''' get optimizer and learning rate scheduler '''
    optim_name = args.optim_name.lower()
    lrschedule = args.lrschedule.lower()

    if optim_name == "sgd":
        optim = SGD(params=model.parameters(),
                    momentum=0.9,
                    lr=args.optim_lr,
                    weight_decay=args.reg_weight)
    elif optim_name == "adam":
        optim = Adam(params=model.parameters(),
                     lr=args.optim_lr,
                     weight_decay=args.reg_weight)
    else:
        optim = AdamW(params=model.parameters(),
                      lr=args.optim_lr,
                      weight_decay=args.reg_weight)

    if lrschedule == "cosine":
        scheduler = CosineAnnealingLR(optim, T_max=args.max_epochs)
    elif lrschedule == "warmupcosine":
        scheduler = WarmupCosineSchedule(optim, t_total=args.max_epochs,
                                         warmup_steps=args.warmup_steps)
    else:
        scheduler = None
    return optim, scheduler


def get_losses(args: ArgumentParser):
    ''' get loss functions '''
    data_range = args.r_max - args.r_min
    ssim = MaskedLoss(func=SSIMLoss(max_val=data_range), 
                      w_out=args.w_out, expand=args.expand)
    grad = GradLoss(square=True, reduction="mean")
    dice = DiceCELoss(n_classes=2, include_bg=True,
                      ce_lambda=2.0, dice_lambda=1.0)
    losses = {
        "sim": {"f": ssim, "w": args.w0_sim},
        "seg": {"f": dice, "w": args.w1_seg},
        "reg": {"f": ssim, "w": args.w2_reg},
        "smooth": {"f": grad, "w": args.w3_smooth},
    }
    return losses


def get_metrics(args: ArgumentParser):
    ''' get metric functions '''
    data_range = args.r_max - args.r_min
    mse = MSEMetric()
    wmse = MaskedMetric(func=mse, w_out=0, expand=1.1)
    dsc = DiceMetric(n_classes=2, ignore_bg=True)
    psnr = PSNRMetric(max_val=data_range)
    ssim = SSIMMetric(max_val=data_range)
    matrix = ConfusionMatrixMetric(n_classes=2, ignore_bg=True)
    metrics = {
        "dice": {"f": dsc},
        "PSNR": {"f": psnr},
        "SSIM": {"f": ssim},
        "MSE-ROI": {"f": mse},        # MSE for 3D-ROI
        "MSE-PN": {"f": wmse},       # MSE for Nodule
        "matrix": {"f": matrix},    # Confusion Matrix
    }
    return metrics


def save_ckpts(model: Module,
               optim: Optimizer,
               args: ArgumentParser,
               save_name: str):
    ''' save checkpoints (model and optimizer states)  '''
    state_dict = {
        "net": model.state_dict(),
        "optimizer": optim.state_dict()
    }
    path = os.path.join(args.model_save_dir, args.pred_model)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, save_name)
    torch.save(state_dict, path)
    print("SAVE checkpoints to `%s`..." % path)


class LogWriter:
    ''' Log Writer Based on Pandas '''

    def __init__(self, save_dir: str, prefix: str = None):
        ''' Args:
        * `save_dir`: save place of log files.
        * `prefix`: prefix-name of the log file.
        '''
        self.data = pd.DataFrame()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        now = time.strftime("%y%m%d%H%M", time.localtime())
        fname = f"{prefix}-{now}.csv" if prefix else f"{now}.csv"
        self.path = os.path.join(save_dir, fname)

    def add_row(self, data: dict):
        temp = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, temp], ignore_index=True)

    def save(self):
        self.data.to_csv(self.path, index=False)
        print("SAVE runtime logs to `%s`..." % self.path)