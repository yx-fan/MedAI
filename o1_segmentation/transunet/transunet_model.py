import torch
import torch.nn as nn

# UNet sample block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ViT encoder block
class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=1, patch_size=16, emb_size=768, img_size=(128, 128)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, E]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_emb
        return x

# Transformer encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size, nhead=heads, dim_feedforward=mlp_dim,
                dropout=dropout, batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# TransUNet body
class TransUNet(nn.Module):
    def __init__(self, in_ch=5, img_size=(256, 256), patch_size=16, emb_size=768, depth=12, heads=12, num_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbedding(in_ch, patch_size, emb_size, img_size)
        self.transformer = TransformerEncoder(emb_size, depth, heads)

        self.up1 = nn.ConvTranspose2d(emb_size, 256, 2, stride=2)
        self.conv1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = ConvBlock(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv3 = ConvBlock(16, 16)
        self.up4 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)           # [B, 1+HW/ps^2, E]
        x = self.transformer(x)
        x = x[:, 1:, :]                   # drop CLS
        h = w = int(x.size(1) ** 0.5)     # e.g., 16x16 patches for 256/16
        x = x.permute(0, 2, 1).contiguous().view(B, -1, h, w)

        x = self.up1(x); x = self.conv1(x)
        x = self.up2(x); x = self.conv2(x)
        x = self.up3(x); x = self.conv3(x)
        x = self.up4(x)
        x = self.head(x)
        return x