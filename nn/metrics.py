import math
import torch
import numpy as np

from typing import Union, Tuple, Sequence, Callable
from torch import nn, Tensor
from torch.nn import functional as F


class MaskedMetric(nn.Module):
    ''' Weighted and Masked Metric '''
    
    def __init__(self, 
                 func: Callable,
                 w_out: float = 0.0, 
                 expand: float = 1.0):
        ''' Args:
        * `func`: basic metric function.
        * `w_out`: weighted coefficient for points outside the ROI.
        * `expand`: expansion coefficient for the ROI.
        '''
        super(MaskedMetric, self).__init__()
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
        metric = self.func(roi * y_pred, roi * y_true)
        return metric


class PSNRMetric(nn.Module):
    ''' Peak Signal-to-Noise Ratio Metric '''

    def __init__(self, max_val: Union[int, float] = 1.0):
        ''' Args:
        * `max_val`: the dynamic range of the inputs.
        '''
        super(PSNRMetric, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"The shape of `y_pred`({y_pred.shape}) and " +
                f"`y_true`({y_true.shape}) should be the same."
            )
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        mse = torch.pow(y_true - y_pred, 2).mean()
        psnr = 20 * math.log10(self.max_val) - 10 * torch.log10(mse)
        return psnr.item()
    

class SSIMMetric(nn.Module):
    ''' Structural Similarity Index Measure Metric '''

    def __init__(self,
                 max_val: Union[int, float] = 1.0,
                 win_size: int = 11,
                 sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03):
        ''' Args:
        * `max_val`: the dynamic range of the inputs.
        * `win_size`: window size of kernel.
        * `sigma`: standard deviation for Gaussian kernel.
        * `k1, k2`: stability constant used in the luminance denominator.
        '''
        super(SSIMMetric, self).__init__()
        self.max_val = max_val
        self.win_size = win_size
        self.sigma = sigma
        self.k1, self.k2 = k1, k2

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
        # compute SSIM
        ss_nr = (2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)
        ss_dr = (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
        ssim = (ss_nr / ss_dr).mean()
        return ssim.item()


class MSEMetric(nn.Module):
    ''' Mean Squared Error Metric '''

    def __init__(self):
        super(MSEMetric, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape != y_true.shape:
            error = f"The shape of `y_pred`({y_pred.shape}) and " +\
                    f"`y_true`({y_true.shape}) should be the same."
            raise ValueError(error)
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        mse = torch.pow(y_true - y_pred, 2).mean()
        return mse.item()
    

class MAEMetric(nn.Module):
    ''' Mean Absolute Error Metric '''
    
    def __init__(self):
        super(MAEMetric, self).__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape != y_true.shape:
            error = f"The shape of `y_pred`({y_pred.shape}) and " +\
                    f"`y_true`({y_true.shape}) should be the same."
            raise ValueError(error)
        if isinstance(y_pred, Tensor): y_pred = y_pred.float()
        if isinstance(y_true, Tensor): y_true = y_true.float()

        mae = torch.abs(y_true - y_pred).mean()
        return mae.item()
    

class DiceMetric(nn.Module):
    ''' Dice Metric of Segmentation Tasks '''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = 1e-5,
                 ignore_bg: bool = False):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `ignore_bg`: whether ignoring background when computering dice.
        '''
        super(DiceMetric, self).__init__()
        self.n_classes = n_classes
        self.cls_start = int(ignore_bg)
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape != y_true.shape:
            error = f"The shape of `y_pred`({y_pred.shape}) and " +\
                    f"`y_true`({y_true.shape}) should be the same."
            raise ValueError(error)

        dice_list = []
        (sm_nr, sm_dr) = self.smooth
        sp_dims = np.arange(2, y_pred.ndim).tolist()
        for i in range(self.cls_start, self.n_classes):
            p, t = y_pred == i, y_true == i
            pt_sum = torch.sum(p * t, dim=sp_dims)
            p_sum = torch.sum(p, dim=sp_dims)
            t_sum = torch.sum(t, dim=sp_dims)
            dice = (2 * pt_sum + sm_nr) / (p_sum + t_sum + sm_dr)
            dice_list.append(dice.item())
        dsc = np.mean(dice_list)
        return dsc
    

class ConfusionMatrixMetric(nn.Module):
    ''' Confusion Matrix Metric '''

    def __init__(self, 
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = 1e-5,
                 ignore_bg: bool = False):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `ignore_bg`: whether ignoring background when computering dice.
        '''
        super(ConfusionMatrixMetric, self).__init__()
        self.n_classes = n_classes
        self.cls_start = int(ignore_bg)
        if isinstance(smooth, (tuple, list)):
            self.smooth = smooth
        else:
            self.smooth = (smooth, smooth)

    def forward(self, 
                y_pred: Union[Tensor, np.ndarray], 
                y_true: Union[Tensor, np.ndarray]):
        if isinstance(y_pred, Tensor): 
            y_pred = y_pred.squeeze(0).squeeze(0).cpu().numpy()
        if isinstance(y_true, Tensor): 
            y_true = y_true.squeeze(0).squeeze(0).cpu().numpy()
        if y_pred.shape != y_true.shape:
            error = f"The shape of `y_pred`({y_pred.shape}) and " +\
                    f"`y_true`({y_true.shape}) should be the same."
            raise ValueError(error)

        tp_list, fp_list, tn_list, fn_list = [], [], [], []
        for i in range(self.cls_start, self.n_classes):
            p1 = (y_pred == i).astype("int16")
            m1 = (y_true == i).astype("int16") 
            m0 = (y_true != i).astype("int16") 
            tp = (p1 + m1) == 2             # true positive
            tn = (p1 + m1) == 0             # true negetive
            fn = m1 - tp.astype("int16")    # false negetive
            fp = m0 - tn.astype("int16")    # false positive
            tp_list.append(np.sum(np.sum(np.sum(tp))))
            tn_list.append(np.sum(np.sum(np.sum(tn))))
            fn_list.append(np.sum(np.sum(np.sum(fn))))
            fp_list.append(np.sum(np.sum(np.sum(fp))))
        matrix = np.stack([tp_list, fp_list, tn_list, fn_list], axis=-1)
        return matrix
    
    def compute(self, method: str, matrix: np.ndarray):
        ''' compute confusion matrix related metric '''
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, ...]
        if matrix.shape[-1] != 4:
            raise ValueError("the last dim of `matrix` should be 4!")
        
        tp = matrix[..., 0].astype("float32")
        fp = matrix[..., 1].astype("float32")
        tn = matrix[..., 2].astype("float32")
        fn = matrix[..., 3].astype("float32")
        p, n = tp + fn, fp + tn
        method = method.lower().replace(" ", "_")
        
        if method in ["tpr", "sen", "recall"]:
            nr, dr = tp, p
        elif method in ["tnr", "spec"]:
            nr, dr = tn, n
        elif method in ["ppv", "precision"]:
            nr, dr = tp, (tp + fp)
        elif method in ["npv"]:
            nr, dr = tn, (tn + fn)
        elif method in ["fnr", "miss_rate"]:
            nr, dr = fn, p
        elif method in ["fpr", "fall_out"]:
            nr, dr = fp, n
        elif method in ["acc"]:
            nr, dr = (tp + tn), (p + n)
        elif method in ["f1"]:
            nr, dr = tp * 2., (tp * 2. + fn + fp)
        else:
            raise ValueError(f"Method `{method}` is not supported!")
        metric = (nr + self.smooth[0]) / (dr + self.smooth[1])
        return metric
