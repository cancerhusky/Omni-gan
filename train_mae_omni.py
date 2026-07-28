import os, yaml, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mae_omni_generator import MAEOmniGenerator
from models.discriminator import ViTDiscriminator
from datasets.sem_dataset import SEMDataset
from utils.wgan_gp import compute_gradient_penalty
from utils.logger import Logger

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mae_omni_sem.yaml")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    logger = Logger(log_dir=os.path.join(cfg["save_dir"], "log"))

    # Dataset
    dataset = SEMDataset(root=cfg["data_root"], img_size=cfg["img_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    # Model Init
    G = MAEOmniGenerator(
        latent_dim=cfg["latent_dim"],
        img_size=cfg["img_size"],
        pretrain_mae_path=cfg["mae_pretrain_path"]
    ).to(device)
    D = ViTDiscriminator(img_size=cfg["img_size"], channels=cfg["channels"]).to(device)

    # Optimizer
    opt_G = torch.optim.AdamW(G.parameters(), lr=cfg["lr_g"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.AdamW(D.parameters(), lr=cfg["lr_d"], betas=(cfg["beta1"], cfg["beta2"]))

    # 感知损失主干（冻结MAE，提取特征）
    mae_percept = G.mae_encoder
    for p in mae_percept.parameters():
        p.requires_grad = False

    for epoch in range(cfg["epochs"]):
        pbar = tqdm(dataloader)
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            b = real_imgs.shape[0]

            # ========= Train Discriminator =========
            opt_D.zero_grad()
            z = torch.randn(b, cfg["latent_dim"]).to(device)
            fake_imgs = G(z).detach()

            d_real = D(real_imgs)
            d_fake = D(fake_imgs)
            gp = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
            loss_D = torch.mean(d_fake) - torch.mean(d_real) + cfg["gp_lambda"] * gp
            loss_D.backward()
            opt_D.step()

            # ========= Train Generator =========
            opt_G.zero_grad()
            z = torch.randn(b, cfg["latent_dim"]).to(device)
            fake_imgs = G(z)
            d_fake_score = D(fake_imgs)
            loss_adv = -torch.mean(d_fake_score)

            # MAE感知损失
            feat_real = mae_percept.forward_features(real_imgs)
            feat_fake = mae_percept.forward_features(fake_imgs)
            loss_percept = torch.nn.functional.mse_loss(feat_real, feat_fake)

            loss_G = loss_adv + cfg["lambda_perceptual"] * loss_percept
            loss_G.backward()
            opt_G.step()

            pbar.set_description(f"E:{epoch} LossD:{loss_D.item():.3f} LossG:{loss_G.item():.3f}")

        # Save samples & checkpoint
        if epoch % cfg["sample_interval"] == 0:
            logger.save_images(fake_imgs, epoch, save_path=os.path.join(cfg["save_dir"], f"sample_{epoch}.png"))
        if epoch % cfg["ckpt_interval"] == 0:
            torch.save({"G":G.state_dict(),"D":D.state_dict(),"epoch":epoch},
                       os.path.join(cfg["save_dir"], f"ckpt_{epoch}.pth"))

if __name__ == "__main__":
    train()
