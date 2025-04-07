import torch
import numpy as np
import SimpleITK as sitk

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader
from utils.img_utils import normalize_ct


def test():
    psnr_list, ssim_list, mse_roi_list, mse_pn_list = [], [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []
    dsc_list, ppv_list, time_list = [], [], []
    test_loader = get_loader(args, is_test=True)
    model = get_pred_model(args, pretrained=True)
    metrics = get_metrics(args)
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        # start test model
        for bid, batch in enumerate(test_loader):
            # predicts growth
            for idx in range(len(batch) - 1):
                batch[idx] = batch[idx].to(args.device)
            (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, info) = batch
            starter.record()
            im2_pred, mk2_pred = model(im0, im1, tm0, tm1)
            ender.record()
            # waiting for GPU sync
            torch.cuda.synchronize()
            time_list.append(starter.elapsed_time(ender) / 1000)
            # computes metrics
            mse_roi = metrics["MSE-ROI"]["f"](im2_pred, im2)
            mse_pn = metrics["MSE-PN"]["f"](im2_pred, im2, mk2)
            psnr = metrics["PSNR"]["f"](im2_pred, im2)
            ssim = metrics["SSIM"]["f"](im2_pred, im2)
            dice = metrics["dice"]["f"](mk2_pred, mk2)
            # computes matrix
            matrix = metrics["matrix"]["f"](mk2_pred, mk2)
            tpr = metrics["matrix"]["f"].compute("tpr", matrix)
            tnr = metrics["matrix"]["f"].compute("tnr", matrix)
            fpr = metrics["matrix"]["f"].compute("fpr", matrix)
            fnr = metrics["matrix"]["f"].compute("fnr", matrix)
            ppv = metrics["matrix"]["f"].compute("ppv", matrix)
            # add into list
            psnr_list.append(np.mean(psnr))
            ssim_list.append(np.mean(ssim))
            mse_roi_list.append(np.mean(mse_roi))
            mse_pn_list.append(np.mean(mse_pn))
            dsc_list.append(np.mean(dice))
            ppv_list.append(np.mean(ppv))
            tpr_list.append(np.mean(tpr))
            fpr_list.append(np.mean(fpr))
            tnr_list.append(np.mean(tnr))
            fnr_list.append(np.mean(fnr))
            # prints information
            save_predictions(im2, mk2, im2_pred, mk2_pred, info)
            print("【Test】Step: %d," % (bid),
                  "PSNR: %.2f, SSIM: %.2f%%," % (psnr_list[-1], ssim_list[-1] * 100),
                  "MSE-ROI: %.2e, MSE-PN: %.2e," % (mse_roi_list[-1], mse_pn_list[-1]),
                  "DSC: %.2f%%, PPV: %.2f%%," % (dsc_list[-1] * 100, ppv_list[-1] * 100),
                  "TPR: %.2f%%, TNR: %.2f%%," % (tpr_list[-1] * 100, tnr_list[-1] * 100),
                  "FPR: %.2f%%, FNR: %.2f%%," % (fpr_list[-1] * 100, fnr_list[-1] * 100),
                  "Time: %.2fs" % (time_list[-1]))
    # compute throughput
    throughput = 1.0 / np.mean(time_list) if len(time_list) > 0 else 0
    # computes means of metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse_roi = np.mean(mse_roi_list)
    avg_mse_pn = np.mean(mse_pn_list)
    avg_dsc = np.mean(dsc_list)
    avg_tpr = np.mean(tpr_list)
    avg_tnr = np.mean(tnr_list)
    avg_fpr = np.mean(fpr_list)
    avg_fnr = np.mean(fnr_list)
    avg_ppv = np.mean(ppv_list)
    print("【%s】Throughput: %.1fimg/s," % (args.pred_model, throughput),
          "avg PSNR: %.2f, avg SSIM: %.2f%%," % (avg_psnr, avg_ssim * 100),
          "avg MSE-ROI: %.2e, avg MSE-PN: %.2e," % (avg_mse_roi, avg_mse_pn),
          "avg DSC: %.2f%%, avg PPV: %.2f%%," % (avg_dsc * 100, avg_ppv * 100),
          "avg TPR: %.2f%%, avg TNR: %.2f%%," % (avg_tpr * 100, avg_tnr * 100),
          "avg FPR: %.2f%%, avg FNR: %.2f%%," % (avg_fpr * 100, avg_fnr * 100))


def save_predictions(im2: Tensor, mk2: Tensor, im2_pred: Tensor, 
                     mk2_pred: Tensor, info: dict):
    ''' save predicted image and mask '''
    SCOPE, RANGE = (args.s_min, args.s_max), (args.r_min, args.r_max)
    pid, nid = info["PatientId"][0], "%02d" % int(info["NoduleId"][0])
    date0, date1, date2 = info["t0"][0], info["t1"][0], info["t2"][0]
    save_dir = f"{args.pred_save_dir}/{args.pred_model}/{pid}/{nid}"
    fname = f"{save_dir}/{date0}+{date1}={date2}"
    os.makedirs(save_dir, exist_ok=True)

    im2_arr = normalize_ct(im2.cpu().numpy(), scope=RANGE, range=SCOPE)
    im2_img = sitk.GetImageFromArray(im2_arr.astype("float32"))
    mk2_img = sitk.GetImageFromArray(mk2.cpu().numpy().astype("uint8"))
    sitk.WriteImage(im2_img, f"{fname}_SrcImg.nii.gz")
    sitk.WriteImage(mk2_img, f"{fname}_SrcMsk.nii.gz")

    im2p_arr = normalize_ct(im2_pred.cpu().numpy(), scope=RANGE, range=SCOPE)
    im2p_img = sitk.GetImageFromArray(im2p_arr.astype("float32"))
    mk2p_img = sitk.GetImageFromArray(mk2_pred.cpu().numpy().astype("uint8"))
    sitk.WriteImage(im2p_img, f"{fname}_PredImg.nii.gz")
    sitk.WriteImage(mk2p_img, f"{fname}_PredMsk.nii.gz")


if __name__ == "__main__":
    test()
