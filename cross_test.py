import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader


def test_models():
    psnr_list, ssim_list, mse_roi_list, mse_pn_list = [], [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []
    dsc_list, ppv_list, throughput_list = [], [], []
    val_writer = LogWriter(args.log_save_dir, prefix=args.pred_model)
    test_loader = get_loader(args, is_test=True)
    metrics = get_metrics(args)
    
    for fd in range(args.num_folds):
        # modify `fold` and `trainded_model`
        args.fold = fd
        args.trained_model = f"best_model_f{fd}.pth"
        # get pretrained model
        model = get_pred_model(args, pretrained=True)
        # compute metrics
        rets = test(fold=fd, model=model, loader=test_loader, 
                    writer=val_writer, metrics=metrics)
        psnr_list.append(rets["PSNR"])
        ssim_list.append(rets["SSIM"])
        mse_roi_list.append(rets["MSE-ROI"])
        mse_pn_list.append(rets["MSE-PN"])
        dsc_list.append(rets["DSC"])        
        ppv_list.append(rets["PPV"])
        tpr_list.append(rets["TPR"])
        tnr_list.append(rets["TNR"])
        fpr_list.append(rets["FPR"])
        fnr_list.append(rets["FNR"])
        throughput_list.append(rets["throughput"])
        val_writer.save()
        del model
        torch.cuda.empty_cache()

    # compute mean metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse_roi = np.mean(mse_roi_list)
    avg_mse_pn = np.mean(mse_pn_list)
    avg_dsc = np.mean(dsc_list)
    avg_ppv = np.mean(ppv_list)
    avg_tpr = np.mean(tpr_list)
    avg_tnr = np.mean(tnr_list)
    avg_fpr = np.mean(fpr_list)
    avg_fnr = np.mean(fnr_list)
    throughput = np.mean(throughput_list)
    # prints information
    print("[PSNR] mean: %.2f, list: %s" % (avg_psnr, psnr_list))
    print("[SSIM] mean: %.2f%%, list: %s" % (avg_ssim * 100, ssim_list))
    print("[MSE-ROI] mean: %.2e, list: %s" % (avg_mse_roi, mse_roi_list))
    print("[MSE-PN] mean: %.2e, list: %s" % (avg_mse_pn, mse_pn_list))
    print("[DSC] mean: %.2f%%, list: %s" % (avg_dsc * 100, dsc_list))
    print("[PPV] mean: %.2f%%, list: %s" % (avg_ppv * 100, ppv_list))
    print("[TPR] mean: %.2f%%, list: %s" % (avg_tpr * 100, tpr_list))
    print("[TNR] mean: %.2f%%, list: %s" % (avg_tnr * 100, tnr_list))
    print("[FPR] mean: %.2f%%, list: %s" % (avg_fpr * 100, fpr_list))
    print("[FNR] mean: %.2f%%, list: %s" % (avg_fnr * 100, fnr_list))
    print("【FINISHED】Method: %s, TotalFolds: %d, Throughput: %.1fimg/s" % 
          (args.pred_model, args.num_folds, throughput))


def test(fold, model, loader, metrics, writer):
    psnr_list, ssim_list, mse_roi_list, mse_pn_list = [], [], [], []
    tpr_list, tnr_list, fnr_list, fpr_list = [], [], [], []
    dsc_list, ppv_list, time_list = [], [], []

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    model.eval()
    with torch.no_grad():
        # start test model
        for bid, batch in enumerate(loader):
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
            # add into logs
            writer.add_row({
                "fold": fold,
                "step": bid,
                "PSNR": round(psnr_list[-1], 3),
                "SSIM": round(ssim_list[-1], 5),
                "MSE-ROI": round(mse_roi_list[-1], 5),
                "MSE-PN": round(mse_pn_list[-1], 6),
                "DSC": round(dsc_list[-1], 5),
                "PPV": round(ppv_list[-1], 5),
                "TPR": round(tpr_list[-1], 5),
                "TNR": round(tnr_list[-1], 5),
                "FPR": round(fpr_list[-1], 5),
                "FNR": round(fnr_list[-1], 5),
            })
            # prints information
            print("【Test】Fold: %d, Step: %d," % (fold, bid),
                  "PSNR: %.2f, SSIM: %.2f%%," % (psnr_list[-1], ssim_list[-1] * 100),
                  "MSE-ROI: %.2e, MSE-PN: %.2e," % (mse_roi_list[-1], mse_pn_list[-1]),
                  "DSC: %.2f%%, PPV: %.2f%%," % (dsc_list[-1] * 100, ppv_list[-1] * 100),
                  "TPR: %.2f%%, TNR: %.2f%%," % (tpr_list[-1] * 100, tnr_list[-1] * 100),
                  "FPR: %.2f%%, FNR: %.2f%%," % (fpr_list[-1] * 100, fnr_list[-1] * 100),
                  "Time: %.2fs" % (time_list[-1]))
     # compute throughput
    throughput = 1.0 / np.mean(time_list) if len(time_list) > 0 else 0
    # computes means of metrics
    avg_psnr = round(np.mean(psnr_list), 5)
    avg_ssim = round(np.mean(ssim_list), 6)
    avg_mse_roi = round(np.mean(mse_roi_list), 6)
    avg_mse_pn = round(np.mean(mse_pn_list), 6)
    avg_dsc = round(np.mean(dsc_list), 6)
    avg_tpr = round(np.mean(tpr_list), 6)
    avg_tnr = round(np.mean(tnr_list), 6)
    avg_fpr = round(np.mean(fpr_list), 6)
    avg_fnr = round(np.mean(fnr_list), 6)
    avg_ppv = round(np.mean(ppv_list), 6)
    print("【Test】Fold: %d, Throughput: %.1fimg/s," % (args.fold, throughput),
          "avg PSNR: %.2f, avg SSIM: %.2f%%," % (avg_psnr, avg_ssim * 100),
          "avg MSE-ROI: %.2e, avg MSE-PN: %.2e," % (avg_mse_roi, avg_mse_pn),
          "avg DSC: %.2f%%, avg PPV: %.2f%%," % (avg_dsc * 100, avg_ppv * 100),
          "avg TPR: %.2f%%, avg TNR: %.2f%%," % (avg_tpr * 100, avg_tnr * 100),
          "avg FPR: %.2f%%, avg FNR: %.2f%%," % (avg_fpr * 100, avg_fnr * 100))
    return {
        "PSNR": avg_psnr,   "SSIM": avg_ssim,
        "MSE-ROI": avg_mse_roi,  "MSE-PN": avg_mse_pn,
        "DSC": avg_dsc,     "PPV": avg_ppv,
        "TPR": avg_tpr,     "TNR": avg_tnr,
        "FPR": avg_fpr,     "FNR": avg_fnr,
        "throughput": throughput,
    }


if __name__ == "__main__":
    test_models()
