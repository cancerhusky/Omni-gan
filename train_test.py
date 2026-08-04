import torch
import torch.nn.functional as F
import run_test
from torch.utils.data import DataLoader


def train_loop(
    generator: run_test.MAEVAEOmniGenerator,
    discriminator,
    mae_encoder,
    dataloader: DataLoader,
    opt_g, opt_d,
    device,
    kl_weight=0.001
):
    generator.train()
    discriminator.train()
    mae_encoder.eval()

    for real_img, cond_vec in dataloader:
        real_img = real_img.to(device)
        cond_vec = cond_vec.to(device)

        with torch.no_grad():
            mae_token_input = mae_encoder(real_img)

        # ---------------- Train D ----------------
        opt_d.zero_grad()
        d_real = discriminator(real_img)
        fake_img, _, _ = generator(mae_token_input, cond_vec)
        d_fake = discriminator(fake_img.detach())

        loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()

        # ---------------- Train G ----------------
        opt_g.zero_grad()
        fake_img, z, kl_loss = generator(mae_token_input, cond_vec)
        d_fake_g = discriminator(fake_img)
        loss_adv_g = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))

        loss_g_total = loss_adv_g + kl_weight * kl_loss
        loss_g_total.backward()
        opt_g.step()

        print(f"D:{loss_d.item():.4f} | G:{loss_g_total.item():.4f} | KL:{kl_loss.item():.4f}")
        