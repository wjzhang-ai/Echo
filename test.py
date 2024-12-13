
from dataloader.Beyound_Dataset import AudioVisualDataset


from utils_tensorboard import *
from utils_criterion import compute_errors
from torchvision import transforms
import time
import os
import numpy as np
import math

import pickle
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from models.allCriterion import *
import hydra
from omegaconf import DictConfig, OmegaConf
from models.functions import *

from models.ecoNet import *


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")

    if cfg.mode.mode != "test":
        raise Exception(
            "This script is for test only. Please run train.py for training"
        )

    # ------------ GPU config ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU, device))

    batch_size = cfg.mode.batch_size

    # ------------ Create dataset -----------

    # Use corresponding dataset class
   
    if cfg.dataset.name == "replica":
        if cfg.mode.eval_on == "val":
            eval_set = AudioVisualDataset("replica", "val")
        else:
            eval_set = AudioVisualDataset("replica", "test")
    elif cfg.dataset.name == "mp3d":
        if cfg.mode.eval_on == "val":
            eval_set = AudioVisualDataset("mp3d", "val")
        else:
            eval_set = AudioVisualDataset("mp3d", "test")

    else:
        raise Exception("Test can be done only on BV1 and BV2")

    print(f"Eval Dataset of {len(eval_set)} instances")
    eval_loader = DataLoader(
        eval_set, batch_size=batch_size, shuffle=False, num_workers=cfg.mode.num_threads
    )

    # ---------- Load Model ----------
   

    model = EcoDepth()
    model.cuda()

    if cfg.mode.criterion == "L1":
        criterion = nn.L1Loss().to(device)
    elif cfg.mode.criterion == "LogDepth":
        criterion= LogDepthLoss()

    if cfg.mode.checkpoints is None:
        raise AttributeError("In test mode, a checkpoint needs to be loaded.")
    else:
        load_epoch = cfg.mode.checkpoints
        experiment_name = "unet_128_mp3d_BS64_Lr0.0001_AdamW_diffusion_test_aspp_l1_deacy0.005"
        checkpoint = torch.load(
            "./checkpoints/"
            + experiment_name
            + "/checkpoint_"
            + str(load_epoch)
            + ".pth"
        )
  
        model.load_state_dict(checkpoint["state_dict"])
        print("Epoch loaded:", str(load_epoch))


    model.eval()

    gt_imgs_to_save = []
    pred_imgs_to_save = []
    rgb_image_to_save=[]
    loss_list = []
    errors = []
    rmse_list = []
    abs_rel_list = []
    log10_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []
    mae_list = []
    i=0
    with torch.no_grad():

        for data in eval_loader:
           
            audio_spec = data["audio_spec"].to(device)
            audio_wave= data["audio_wave"].to(device)
            image=data["img"].to(device)
            # [1, 2, 128, 128]
            
            depthgt = data["depth"].to(device)
           
            depth_pred = model(audio_spec,audio_wave)
   

            loss_test = criterion(depth_pred[depthgt != 0], depthgt[depthgt != 0])
            loss_list.append(loss_test.cpu().item())

            for idx in range(depth_pred.shape[0]):
                gt_imgs_to_save.append(depthgt[idx].detach().cpu().numpy()) 
                pred_imgs_to_save.append(depth_pred[idx].detach().cpu().numpy())
                rgb_image_to_save.append(image[idx].detach().cpu().numpy())
                if cfg.dataset.depth_norm:
                    unscaledgt = (
                        depthgt[idx].detach().cpu().numpy()
                    )
                    unscaledpred = (
                        depth_pred[idx].detach().cpu().numpy() 
                    )
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(
                        unscaledgt, unscaledpred
                    )
                else:
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(
                        depthgt[idx].cpu().numpy(), depth_pred[idx].cpu().numpy()
                    )
                errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))

      

            rmse_list.append(rmse)
            abs_rel_list.append(abs_rel)
            log10_list.append(log_10)
            delta1_list.append(a1)
            delta2_list.append(a2)
            delta3_list.append(a3)
            mae_list.append(mae)

        mean_errors = np.array(errors).mean(0)
        print("abs rel: {:.3f}".format(mean_errors[0]))
        print("RMSE: {:.3f}".format(mean_errors[1]))
        print("Delta1: {:.3f}".format(mean_errors[2]))
        print("Delta2: {:.3f}".format(mean_errors[3]))
        print("Delta3: {:.3f}".format(mean_errors[4]))
        print("Log10: {:.3f}".format(mean_errors[5]))
        print("MAE: {:.3f}".format(mean_errors[6]))

    # Save evaluation
    d = {
        "loss": loss_list,
        "abs_rel": abs_rel_list,
        "rmse": rmse_list,
        "log10": log10_list,
        "delta1": delta1_list,
        "delta2": delta2_list,
        "delta3": delta3_list,
        "mae": mae_list,
        "gt_images": gt_imgs_to_save,
        "pred_imgs": pred_imgs_to_save,
    }


    stats_df = pd.DataFrame(data=d)
    if cfg.mode.eval_on == "test":
         stats_df.to_pickle("test12.pkl")
    else:
        stats_df.to_pickle(
            os.path.join(
                cfg.mode.stat_dir + cfg.dataset.name,
                "val",
                "stats_on_"
                + cfg.dataset.name
                + "_val_set_"
                + cfg.mode.experiment_name
                + "_epoch_"
                + str(load_epoch)
                + ".pkl",
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Exception happened during test")
