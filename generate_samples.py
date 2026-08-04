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


@torch.no_grad()
def interpolate_z(mae_token_a, mae_token_b, cond_a, cond_b, alpha=0.5, generator, device):
    generator.eval()
    _, sem_a = generator.mae_semantic(mae_token_a)
    _, sem_b = generator.mae_semantic(mae_token_b)

    cond_emb_a = generator.cond_projector(cond_a)
    cond_emb_b = generator.cond_projector(cond_b)
    sem_a = sem_a + cond_emb_a
    sem_b = sem_b + cond_emb_b

    mu_a, logvar_a = generator.vae.mu_head(sem_a), generator.vae.logvar_head(sem_a)
    mu_b, logvar_b = generator.vae.mu_head(sem_b), generator.vae.logvar_head(sem_b)

    mu_interp = alpha * mu_a + (1.0 - alpha) * mu_b
    logvar_interp = alpha * logvar_a + (1.0 - alpha) * logvar_b

    std = torch.exp(0.5 * logvar_interp)
    eps = torch.randn_like(std)
    z_interp = mu_interp + eps * std

    rec_sem_interp = generator.vae.z_decoder(z_interp)
    token_sem_a, _ = generator.mae_semantic(mae_token_a)
    fake_interp = generator.decoder(token_sem_a, rec_sem_interp)
    return fake_interp
