import torch
import torch.nn as nn
import torch.nn.functional as F


# Token Transformer Block
class TokenTransBlock(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# MAE as semantic prior
class MAESemanticPrior(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.token_trans = TokenTransBlock(dim=embed_dim)
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )

    def forward(self, mae_token_feat):
        """
        Args:
            mae_token_feat: [B, 256, 768]
        Returns:
            token_semantic: [B,256,768]
            semantic_latent: [B,512]
        """
        token_semantic = self.token_trans(mae_token_feat)
        global_semantic = token_semantic.mean(dim=1)
        semantic_latent = self.semantic_proj(global_semantic)
        return token_semantic, semantic_latent


# VAE: learn generative latent z from semantic latent
class SemanticVAE(nn.Module):
    def __init__(self, semantic_dim=512, latent_z_dim=128):
        super().__init__()
        self.mu_head = nn.Linear(semantic_dim, latent_z_dim)
        self.logvar_head = nn.Linear(semantic_dim, latent_z_dim)
        self.z_decoder = nn.Sequential(
            nn.Linear(latent_z_dim, semantic_dim),
            nn.GELU(),
            nn.LayerNorm(semantic_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, semantic_latent):
        mu = self.mu_head(semantic_latent)
        logvar = self.logvar_head(semantic_latent)
        z = self.reparameterize(mu, logvar)
        rec_semantic = self.z_decoder(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z, rec_semantic, kl_loss


# TransUpDecoder: restore image detail
class TransUpDecoder(nn.Module):
    def __init__(self, embed_dim=768, semantic_dim=512, out_channels=1):
        super().__init__()
        self.z_fusion = nn.Linear(semantic_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim * 4)

        self.upsample1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(embed_dim // 4, embed_dim // 4, 3, padding=1)

        self.upsample2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(embed_dim // 16, embed_dim // 16, 3, padding=1)

        self.upsample3 = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(embed_dim // 64, embed_dim // 64, 3, padding=1)

        self.upsample4 = nn.PixelShuffle(2)
        self.out_conv = nn.Conv2d(embed_dim // 256, out_channels, 3, padding=1)

    def forward(self, token_semantic, recover_semantic):
        B, N, C = token_semantic.shape
        z_feat = self.z_fusion(recover_semantic).unsqueeze(1)
        token_final = token_semantic + z_feat

        x = self.proj(token_final)
        x = x.transpose(1, 2).reshape(B, C * 4, 16, 16)

        x = self.upsample1(x)
        x = F.relu(self.conv1(x))

        x = self.upsample2(x)
        x = F.relu(self.conv2(x))

        x = self.upsample3(x)
        x = F.relu(self.conv3(x))

        x = self.upsample4(x)
        out = torch.tanh(self.out_conv(x))
        return out


# Full generator: condition embedded inside model
class MAEVAEOmniGenerator(nn.Module):
    def __init__(self, cond_input_dim=2):
        """
        cond_input_dim: condition vector dimension, here Y2O3 + magnification = 2
        """
        super().__init__()
        self.mae_semantic = MAESemanticPrior()
        self.vae = SemanticVAE()
        self.decoder = TransUpDecoder()

        # internal condition projector
        self.cond_projector = nn.Sequential(
            nn.Linear(cond_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512)
        )

    def forward(self, mae_token_input, cond_vec):
        """
        Args:
            mae_token_input: [B,256,768], MAE encoder output tokens
            cond_vec: [B, 2], (Y2O3_wt, magnification)
        Returns:
            fake_img: generated image
            z: vae latent vector
            kl_loss: vae kl divergence loss
        """
        token_sem, sem_latent = self.mae_semantic(mae_token_input)

        # project condition and fuse into semantic latent
        cond_emb = self.cond_projector(cond_vec)  # [B,512]
        sem_latent = sem_latent + cond_emb

        z, rec_sem, kl_loss = self.vae(sem_latent)
        fake_img = self.decoder(token_sem, rec_sem)
        return fake_img, z, kl_loss


# ------------------------------
# simple test
if __name__ == "__main__":
    B = 2
    mae_tokens = torch.randn(B, 256, 768)
    cond = torch.randn(B, 2)

    gen = MAEVAEOmniGenerator(cond_input_dim=2)
    img, z, kl = gen(mae_tokens, cond)

    print(f"fake image shape: {img.shape}")
    print(f"latent z shape: {z.shape}")
    print(f"kl loss: {kl.item():.4f}")
