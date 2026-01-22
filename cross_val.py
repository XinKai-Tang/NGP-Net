import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader


def cross_val():
    val_writer = LogWriter(args.log_save_dir, prefix=args.pred_model)
    losses = get_losses(args)
    metrics = get_metrics(args)

    all_time = time.time()
    max_ssim, num_images = [], []
    for fd in range(args.num_folds):
        args.fold = fd      # modify `fold`
        tra_loader, val_loader = get_loader(args, is_test=False)
        model = get_pred_model(args, pretrained=False)
        optim, scheduler = get_optim(model, args)

        max_ssim.append(0)
        num_images.append(len(val_loader))
        fd_time = time.time()
        for ep in range(args.max_epochs):
            loss = run_epoch(fold=fd, epoch=ep, model=model, 
                             loader=tra_loader, optim=optim, losses=losses)
            if scheduler is not None:
                scheduler.step()
            if (ep + 1) % args.val_freq == 0:
                accs = val_epoch(fold=fd, epoch=ep, model=model, 
                                 loader=val_loader, metrics=metrics)
                val_writer.add_row({
                    "fold": fd,
                    "epoch": ep,
                    "train_loss": loss,
                    "val_dice": accs["dice"],
                    "val_psnr": accs["psnr"],
                    "val_ssim": accs["ssim"],
                })
                val_writer.save()
                if accs["ssim"] > max_ssim[-1]:
                    print("【Val】New Best Acc: %f -> %f" % (max_ssim[-1], accs["ssim"]))
                    save_ckpts(model, optim, args, save_name=f"best_model_f{fd}.pth")
                max_ssim[-1] = max(max_ssim[-1], accs["ssim"])
            save_ckpts(model, optim, args, save_name="latest_model.pth")
        print("【Cross Val】Fold: %d, Best Acc: %f," % (args.fold, max_ssim[-1]),
              "Time: %.2fh" % ((time.time() - fd_time) / 3600))
        del tra_loader, val_loader, model, optim, scheduler
        torch.cuda.empty_cache()

    avg_ssim = np.multiply(np.array(num_images), np.array(max_ssim))
    avg_ssim = np.sum(avg_ssim) / np.sum(num_images)
    print("【Best Acc】", [round(a, 4) for a in max_ssim])
    print("【FINISHED】Mean Acc: %f, " % (avg_ssim),
          "Time: %.2fh" % ((time.time() - all_time) / 3600))


def run_epoch(fold, epoch, model, loader, optim, losses):
    model.train()
    ep_time = time.time()
    loss_list = []
    for bid, batch in enumerate(loader):
        start = time.time()
        for idx in range(len(batch) - 1):   # moves to target device
            batch[idx] = batch[idx].to(args.device)
        (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, _) = batch
        # predicts growth
        t_zero = torch.zeros_like(tm1).to(tm1)
        im1_pred0, mk1_pred, _ = model.ngpnet(im0, im0, t_zero, tm1)
        im1_pred1, mk1_pred, _ = model.ngpnet(im0, im1, tm0, t_zero)
        im2_pred, mk2_pred, field2 = model.ngpnet(im0, im1, tm0, tm1)
        # computes losses for NGPnet
        Lsim = losses["sim"]["f"](im2_pred, im2, mk1) * losses["sim"]["w"]
        Lreg = (losses["reg"]["f"](im1_pred1, im1, mk1) +
                losses["reg"]["f"](im1_pred0, im1, mk0)) * losses["reg"]["w"]
        Lseg = losses["seg"]["f"](mk2_pred, mk2) * losses["seg"]["w"]
        Lsmooth = losses["smooth"]["f"](field2) * losses["smooth"]["w"]
        loss = Lsim + Lreg + Lseg + Lsmooth
        loss_list.extend([loss.item()] * len(tm0))
        # backwards losses and prints information
        print("【Train】Fold: %d, Epoch: %d, Batch: %d," % (fold, epoch, bid),
              "Loss: %.4f (sim %.4f," % (loss_list[-1], Lsim.item()),
              "seg %.4f, reg %.4f," % (Lseg.item(), Lreg.item()),
              "smooth %.4f), Time: %.2fs" % (Lsmooth.item(), time.time()-start))
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.empty_cache()
    # computes the mean of losses
    avg_loss = np.mean(loss_list)
    print("【Train】Fold: %d, Epoch: %d, Avg Loss: %f, Time: %.2fs"
          % (fold, epoch, avg_loss, time.time() - ep_time))
    return avg_loss


def val_epoch(fold, epoch, model, loader, metrics):
    model.eval()
    ep_time = time.time()
    ssim_list, psnr_list, dice_list = [], [], []
    with torch.no_grad():
        for bid, batch in enumerate(loader):
            start = time.time()
            for idx in range(len(batch) - 1):   # moves to target device
                batch[idx] = batch[idx].to(args.device)
            (im0, im1, im2, mk0, mk1, mk2, tm0, tm1, _) = batch
            # predicts growth
            im2_pred, mk2_pred = model(im0, im1, tm0, tm1)
            # computes metrics
            Mpsnr = metrics["PSNR"]["f"](im2_pred, im2)
            Mssim = metrics["SSIM"]["f"](im2_pred, im2)
            Mdice = metrics["dice"]["f"](mk2_pred, mk2)
            psnr_list.append(np.mean(Mpsnr))
            ssim_list.append(np.mean(Mssim))
            dice_list.append(np.mean(Mdice))
            # prints information
            print("【Val】Fold: %d, Epoch: %d, Batch: %d," % (fold, epoch, bid),
                  "Dice: %.4f, PSNR: %.4f," % (dice_list[-1], psnr_list[-1]),
                  "SSIM: %.4f," % (ssim_list[-1]),
                  "Time: %.2fs" % (time.time() - start))
    # computes means of metrics
    avg_dice = np.mean(dice_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print("【Val】Fold: %d, Epoch: %d, Avg Dice: %f," % (fold, epoch, avg_dice),
          "Avg PSNR: %f, Avg SSIM: %f," % (avg_psnr, avg_ssim),
          "Time: %.2fs" % (time.time() - ep_time))
    return {
        "dice": avg_dice,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }


if __name__ == "__main__":
    cross_val()
