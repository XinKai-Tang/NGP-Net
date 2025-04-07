import time
import torch
import numpy as np

from utils.common import *
from utils.config import args
from utils.dataloader import get_loader


def train():
    tra_loader, val_loader = get_loader(args, is_test=False)
    model = get_pred_model(args, pretrained=False)
    optim, scheduler = get_optim(model, args)
    losses = get_losses(args)
    metrics = get_metrics(args)
    val_writer = LogWriter(args.log_save_dir, prefix=args.pred_model)

    max_dice, max_ssim, all_time = 0, 0, time.time()
    for ep in range(args.max_epochs):
        loss = run_epoch(ep, model, tra_loader, optim, losses)
        if scheduler is not None:
            scheduler.step()
        if (ep + 1) % args.val_freq == 0:
            accs = val_epoch(ep, model, val_loader, metrics)
            val_writer.add_row({
                "epoch": ep + 1,
                "train_loss": loss,
                "val_dice": accs["dice"],
                "val_psnr": accs["psnr"],
                "val_ssim": accs["ssim"],
            })
            val_writer.save()
            if accs["ssim"] > max_ssim:
                print("【Val】New Best Acc: %f -> %f" % (max_ssim, accs["ssim"]))
                save_ckpts(model, optim, args, save_name="best_model.pth")
            max_dice = max(max_dice, accs["dice"])
            max_ssim = max(max_ssim, accs["ssim"])
        save_ckpts(model, optim, args, save_name="latest_model.pth")
    print("【FINISHED】Best Acc: %f, " % (max_ssim),
          "Time: %.2fh" % ((time.time() - all_time) / 3600))


def run_epoch(epoch, model, loader, optim, losses):
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
        print("【Train】Epoch: %d, Batch: %d," % (epoch, bid),
              "Loss: %.4f (sim %.4f," % (loss_list[-1], Lsim.item()),
              "seg %.4f, reg %.4f," % (Lseg.item(), Lreg.item()),
              "smooth %.4f), Time: %.2fs" % (Lsmooth.item(), time.time()-start))
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.empty_cache()
    # computes the mean of losses
    avg_loss = np.mean(loss_list)
    print("【Train】Epoch: %d, Avg Loss: %f, Time: %.2fs"
          % (epoch, avg_loss, time.time() - ep_time))
    return avg_loss


def val_epoch(epoch, model, loader, metrics):
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
            print("【Val】Epoch: %d, Batch: %d," % (epoch, bid),
                  "Dice: %.4f, PSNR: %.4f," % (dice_list[-1], psnr_list[-1]),
                  "SSIM: %.4f," % (ssim_list[-1]),
                  "Time: %.2fs" % (time.time() - start))
    # computes means of metrics
    avg_dice = np.mean(dice_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print("【Val】Epoch: %d, Avg Dice: %f," % (epoch, avg_dice),
          "Avg PSNR: %f, Avg SSIM: %f," % (avg_psnr, avg_ssim),
          "Time: %.2fs" % (time.time() - ep_time))
    return {
        "dice": avg_dice,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
    }


if __name__ == "__main__":
    train()
