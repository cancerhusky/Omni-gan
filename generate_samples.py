import os, argparse
import torch
from models.mae_omni_generator import MAEOmniGenerator
import yaml
from torchvision.utils import save_image

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mae_omni_sem.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--out_dir", default="./virtual_sem_output")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    G = MAEOmniGenerator(
        latent_dim=cfg["latent_dim"],
        img_size=cfg["img_size"],
        pretrain_mae_path=None
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    with torch.no_grad():
        for idx in range(args.num_samples):
            z = torch.randn(1, cfg["latent_dim"]).to(device)
            fake = G(z)
            save_image(fake, os.path.join(args.out_dir, f"virtual_sem_{idx:04d}.png"), normalize=True)
    print(f"Generate {args.num_samples} virtual SEM images finished!")

if __name__ == "__main__":
    generate()
