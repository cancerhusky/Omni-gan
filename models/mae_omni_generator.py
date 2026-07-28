import torch
import torch.nn as nn
from .mae_vit import MAE_ViT_Encoder

class TransUpBlock(nn.Module):
    """上采样模块，将token特征恢复为图像分辨率"""
    def __init__(self, dim, out_channels=1):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(dim//4, dim//4, 3, padding=1)
        self.out_conv = nn.Conv2d(dim//4, out_channels, 3, padding=1)

    def forward(self, token_feat, H, W):
        B, N, C = token_feat.shape
        token_feat = self.proj(token_feat)
        token_feat = token_feat.transpose(1,2).reshape(B, C*4, H, W)
        x = self.pixel_shuffle(token_feat)
        x = torch.relu(self.conv(x))
        img = torch.tanh(self.out_conv(x))
        return img


class MAEOmniGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        mae_embed_dim=768,
        token_num=256,
        img_size=256,
        patch_size=16,
        pretrain_mae_path=None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_num = token_num
        self.mae_embed_dim = mae_embed_dim

        # 噪声投影：z -> token序列
        self.z_proj = nn.Linear(latent_dim, token_num * mae_embed_dim)

        # MAE预训练ViT Encoder
        self.mae_encoder = MAE_ViT_Encoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=mae_embed_dim, depth=12, num_heads=12
        )
        if pretrain_mae_path is not None:
            self.mae_encoder = MAE_ViT_Encoder.load_mae_pretrain(pretrain_mae_path,
                img_size=img_size, patch_size=patch_size,
                embed_dim=mae_embed_dim, depth=12, num_heads=12
            )

        # 上采样解码器
        self.up_decoder = TransUpBlock(dim=mae_embed_dim, out_channels=1)

    def forward(self, z):
        B = z.shape[0]
        # Step1 噪声映射token
        tokens = self.z_proj(z)
        tokens = tokens.view(B, self.token_num, self.mae_embed_dim)

        # Step2 MAE Encoder提取全局特征【无掩码！方案2核心】
        feat_tokens = self.mae_encoder.blocks(tokens)
        feat_tokens = self.mae_encoder.norm(feat_tokens)
        feat_tokens = feat_tokens[:, 1:, :] # discard cls token

        # Step3 上采样重建图像
        H = W = int(self.token_num**0.5)
        fake_img = self.up_decoder(feat_tokens, H, W)
        return fake_img
