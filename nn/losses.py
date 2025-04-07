import math
import torch
import numpy as np

from typing import Union, Tuple, Sequence, Callable
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, functional as F


class MaskedLoss(nn.Module):
    ''' Weighted and Masked Loss '''
    
    def __init__(self, 
                 func: Callable,
                 w_out: float = 0.0, 
                 expand: float = 1.0):
        ''' Args:
        * `func`: basic loss function.
        * `w_out`: weighted coefficient for points outside the ROI.
        * `expand`: expansion coefficient for the ROI.
        '''
        super(MaskedLoss, self).__init__()
        self.func = func
        self.w_out = w_out
        self.expand = expand
    
    def _compute_roi_(self, mask: Tensor):
        array = mask.cpu().numpy()
        roi_box = np.ones_like(array) * self.w_out

        for b in range(mask.shape[0]):
            gts = np.where(array[b, 0, ...] > 0)
            if gts[0].shape[0] == 0: continue
            bbox, shape = [], mask.shape[2:]
            for gt, sh in zip(gts, shape):
                centroid = (gt.max() + gt.min()) / 2
                edge = (gt.max() - gt.min() + 1) / 2 * self.expand
                bbox.append([int(max(0, centroid - edge)), 
                             int(min(sh-1, centroid + edge))])
            if mask.ndim == 4:
                roi_box[b, 0, 
                        bbox[0][0] : bbox[0][1],
                        bbox[1][0] : bbox[1][1]] = 1.0
            else:
                roi_box[b, 0, 
                        bbox[0][0] : bbox[0][1],
                        bbox[1][0] : bbox[1][1],
                        bbox[2][0] : bbox[2][1]] = 1.0
                
        roi_box = torch.tensor(roi_box, device=mask.device)
        return roi_box

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor):
        roi = self._compute_roi_(mask)
        loss = self.func(roi * y_pred, roi * y_true)
        return loss


class SSIMLoss(nn.Module):
    ''' Structural Similarity Index Measure Loss '''

    def __init__(self,
                 max_val: Union[int, float] = 1.0,
                 win_size: int = 11,
                 sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 reduction: str = "mean"):
        ''' Args:
        * `max_val`: the dynamic range of the inputs.
        * `win_size`: window size of kernel.
        * `sigma`: standard deviation for Gaussian kernel.
        * `k1, k2`: stability constant used in the luminance denominator.
        * `reduction`: reduction function of the loss.
        '''
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.win_size = win_size
        self.sigma = sigma
        self.k1, self.k2 = k1, k2
        self.reduction = reduction

    def _gaussian_kernel_(self,
                          channels: int,
                          window: Sequence[int],
                          sigma: Sequence[float]):
        ''' computing 2D or 3D gaussian kernel '''
        def gaussian_1d(_win: int, _sigma: float):
            ''' computing 1D gaussian kernel '''
            dist = torch.arange((1 - _win) / 2, (1 + _win) / 2, step=1)
            gauss = torch.exp(-torch.pow(dist / _sigma, 2) / 2)
            return (gauss / gauss.sum()).unsqueeze(dim=0)
        
        assert len(sigma) in [2, 3] and len(window) == len(sigma)
        gks = [gaussian_1d(w, s) for w, s in zip(window, sigma)]
        kernel = torch.matmul(gks[0].t(), gks[1])   # (W,1) @ (1,W)
        if len(gks) == 3:
            kernel = torch.mul(
                kernel.unsqueeze(-1).repeat(1, 1, window[2]),
                gks[2][None,].expand(*window)
            )
        kernel = kernel.expand([channels, 1, *window]).contiguous()
        return kernel

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape != y_true.shape:
            error = f"The shape of `y_pred`({y_pred.shape}) and " +\
                    f"`y_true`({y_true.shape}) should be the same."
            raise ValueError(error)
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        channels = y_pred.size(1)               # channels
        sp_dims = y_pred.ndim - 2               # spatial dimensions
        Conv = getattr(F, f"conv{sp_dims}d")    # conv func
        window = [self.win_size] * sp_dims      # window shape
        sigma = [self.sigma] * sp_dims          # kernel sigma
        kernel = self._gaussian_kernel_(channels, window, sigma).to(y_pred)
        
        # compute means
        mu_p = Conv(y_pred, kernel, groups=channels)
        mu_t = Conv(y_true, kernel, groups=channels)
        mu_p2 = Conv(y_pred * y_pred, kernel, groups=channels)
        mu_t2 = Conv(y_true * y_true, kernel, groups=channels)
        mu_pt = Conv(y_pred * y_true, kernel, groups=channels)
        # compute variances and covariance
        var_p = mu_p2 - mu_p * mu_p
        var_t = mu_t2 - mu_t * mu_t
        cov_pt = mu_pt - mu_p * mu_t
        # compute stability constants for luminance/contrast
        c1 = (self.k1 * self.max_val) ** 2  # for luminance
        c2 = (self.k2 * self.max_val) ** 2  # for contrast
        # compute SSIM Loss
        ss_nr = (2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)
        ss_dr = (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
        loss = 1.0 - (ss_nr / ss_dr)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class MSELoss(nn.Module):
    ''' Mean Squared Error Loss '''
    
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        loss = torch.mean((y_true - y_pred) ** 2)
        return loss


class GradLoss(nn.Module):
    ''' Gradient Loss '''
    
    def __init__(self, 
                 square: bool = True,
                 reduction: str = "mean"):
        ''' Args:
        * `square`: whether using squared diffusion result.
        * `reduction`: reduction function of the loss.
        '''
        super(GradLoss, self).__init__()
        self.square = square
        self.reduction = reduction

    def _diff_(self, x: Tensor):
        sp_dims = x.ndim - 2    # spatial dimensions
        df = [None] * sp_dims
        for i in range(sp_dims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, sp_dims + 2)]
            x = x.permute(r)
            df_i = x[1:, ...] - x[:-1, ...]
            # permute back
            r = [*range(d - 1, d + 1), *range(d - 2, 0, -1), 
                 0, *range(d + 1, sp_dims + 2)]
            df[i] = df_i.permute(r)
        return df

    def forward(self, y_pred: Tensor, y_true: Tensor = None):
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        if not self.square:     # l1
            df = [torch.abs(f) for f in self._diff_(y_pred)]
        else:                   # l2
            df = [(f * f) for f in self._diff_(y_pred)]
        df = [f.flatten(start_dim=1).mean(dim=-1) for f in df]
        loss = sum(df) / len(df)
        
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class DiceLoss(nn.Module):
    ''' Dice Loss for Segmentation Tasks'''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = True,
                 reduction: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `reduction`: reduction function of dice loss.
        '''
        super(DiceLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                "`reduction` of dice loss should be 'mean' or 'sum'!"
            )

        self.n_classes = n_classes
        self.sigmoid_x = sigmoid_x
        self.softmax_x = softmax_x
        self.onehot_y = onehot_y
        self.square_xy = square_xy
        self.include_bg = include_bg
        self.reduction = reduction
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def forward(self, pred: Tensor, mask: Tensor):
        (sm_nr, sm_dr) = self.smooth

        if self.sigmoid_x:
            pred = torch.sigmoid(pred)
        if self.n_classes > 1:
            if self.softmax_x and self.n_classes == pred.size(1):
                pred = torch.softmax(pred, dim=1)
            if self.onehot_y:
                mask = mask if mask.ndim < 5 else mask.squeeze(dim=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute(0, 4, 1, 2, 3).float()
            if not self.include_bg:     # ignore background class
                pred = pred[:, 1:] if pred.size(1) > 1 else pred
                mask = mask[:, 1:] if mask.size(1) > 1 else mask
        if pred.ndim != mask.ndim or pred.size(1) != mask.size(1):
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`mask`({mask.shape}) should be the same."
            )

        # only reducing spatial dimensions:
        reduce_dims = torch.arange(2, pred.ndim).tolist()
        insersect = torch.sum(pred * mask, dim=reduce_dims)
        if self.square_xy:
            pred, mask = torch.pow(pred, 2), torch.pow(mask, 2)
        pred_sum = torch.sum(pred, dim=reduce_dims)
        mask_sum = torch.sum(mask, dim=reduce_dims)
        loss = 1. - (2 * insersect + sm_nr) / (pred_sum + mask_sum + sm_dr)

        if self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    ''' Dice Loss with Cross Entropy Loss '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = False,
                 dice_reduct: str = "mean",
                 ce_weight: Tensor = None,
                 ce_reduct: str = "mean",
                 dice_lambda: float = 1.0,
                 ce_lambda: float = 1.0):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg when computering dice.
        * `dice_reduct`: reduction function of dice loss.
        * `ce_weight`: weight of cross entropy loss.
        * `ce_reduct`: reduction function of cross entropy loss.
        * `dice_lambda`: weight coef of dice loss in total loss.
        * `ce_lambda`: weight coef of cross entropy loss in total loss.
        '''
        super(DiceCELoss, self).__init__()
        if dice_lambda < 0:
            raise ValueError(
                f"`dice_lambda` should be no less than 0, but got {dice_lambda}."
            )
        if ce_lambda < 0:
            raise ValueError(
                f"`ce_lambda` should be no less than 0, but got {ce_lambda}."
            )

        self.dice_lambda = dice_lambda
        self.ce_lambda = ce_lambda
        self.dice_loss = DiceLoss(n_classes=n_classes,
                                  smooth=smooth,
                                  sigmoid_x=sigmoid_x,
                                  softmax_x=softmax_x,
                                  onehot_y=onehot_y,
                                  square_xy=square_xy,
                                  include_bg=include_bg,
                                  reduction=dice_reduct)
        self.ce_loss = CrossEntropyLoss(weight=ce_weight,
                                        reduction=ce_reduct)

    def cross_entropy(self, pred: Tensor, mask: Tensor):
        # reducing the channel dimension:
        if pred.size(1) == mask.size(1):
            mask = mask.argmax(dim=1)   # one-hot format
        else:
            mask = mask.squeeze(dim=1)  # (B,C,H,W,D) format
        return self.ce_loss(pred, mask.long())

    def forward(self, pred: Tensor, mask: Tensor):
        dice_loss = self.dice_loss(pred, mask) * self.dice_lambda
        ce_loss = self.cross_entropy(pred, mask) * self.ce_lambda
        return (dice_loss + ce_loss)
    