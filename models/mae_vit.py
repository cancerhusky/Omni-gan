import torch
from timm.models.vision_transformer import VisionTransformer

class MAE_ViT_Encoder(VisionTransformer):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0, # 移除分类头
            global_pool=False
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x # [B, num_token, embed_dim]

    @classmethod
    def load_mae_pretrain(cls, weight_path, **kwargs):
        model = cls(**kwargs)
        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = {}
        for k,v in ckpt["model"].items():
            if "encoder." in k:
                new_k = k.replace("encoder.", "")
                if new_k in model.state_dict():
                    state_dict[new_k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("MAE Encoder load pretrain:", msg)
        return model
