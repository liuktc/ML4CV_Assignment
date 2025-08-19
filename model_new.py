import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 1. Frozen DINOv2 wrapper
# -------------------------------
class FrozenDINOv2(nn.Module):
    def __init__(self, dinov2_model, out_dim=None):
        super().__init__()
        self.dino = dinov2_model
        for p in self.dino.parameters():
            p.requires_grad = False
        self.out_dim = out_dim

        if out_dim is not None:
            self.proj = nn.Conv2d(self.dino.embed_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Assumes dinov2_model returns patch embeddings (B, num_patches, C)
        We'll reshape to (B, C, H_p, W_p).
        """
        B, C, H, W = x.shape
        H = H // 16
        W = W // 16
        feats = self.dino(x, is_training=True)["x_norm_patchtokens"]
        B, HW, d_model = feats.shape
        feats = feats.view((B, H, W, d_model))
        feats = torch.permute(feats, (0, 3, 1, 2))
        feats = self.proj(feats)  # (B, out_dim, H_p, W_p)

        return feats


# -------------------------------
# 2. Small CNN encoder for local details
# -------------------------------
class LocalCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, down_scaling_factor=1):
        super().__init__()
        # self.down_scaling_factor = down_scaling_factor

        if down_scaling_factor == 1:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding="same", stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding="same", stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_ch, 3, padding="same"),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif down_scaling_factor == 2:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif down_scaling_factor == 4:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.encoder(
            x
        )  # (B, out_ch, H/down_scaling_factor, W/down_scaling_factor)


# -------------------------------
# 3. Decoder + projection head
# -------------------------------
class DecoderHead(nn.Module):
    def __init__(self, in_ch, embed_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, 1),  # projection to embedding dim
        )

    def forward(self, x):
        return self.decoder(x)


# class DecoderHead(nn.Module):
#     def __init__(self, in_ch, embed_dim=128):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # predict features for PixelShuffle (needs out_ch * r^2 channels)
#             nn.Conv2d(256, embed_dim * (4 * 4), kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor=4),  # rearranges channels into 4×H, 4×W
#             nn.BatchNorm2d(embed_dim),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.decoder(x)


# -------------------------------
# 4. Full model
# -------------------------------
class DinoMetricLearning(nn.Module):
    def __init__(
        self,
        dinov2_model,
        dino_out_dim=384,
        cnn_out_dim=128,
        embed_dim=128,
        down_scaling_factor=1,
        normalize=True,
    ):
        super().__init__()
        self.dino = FrozenDINOv2(dinov2_model, out_dim=dino_out_dim)
        self.local_cnn = LocalCNN(
            in_ch=3, out_ch=cnn_out_dim, down_scaling_factor=down_scaling_factor
        )
        self.decoder = DecoderHead(dino_out_dim + cnn_out_dim, embed_dim)
        self.normalize = normalize

    def forward(self, x):
        B, _, H, W = x.shape

        # Local CNN features (high-res, e.g., H/4)
        local_feats = self.local_cnn(
            x
        )  # (B, C_c, H / down_scaling_factor, W / down_scaling_factor)

        # DINO features (low-res, e.g., H/14)
        dino_feats = self.dino(x)  # (B, C_d, H_p, W_p)
        dino_up = F.interpolate(
            dino_feats, size=local_feats.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate
        fused = torch.cat(
            [dino_up, local_feats], dim=1
        )  # (B, C_d+C_c, H / down_scaling_factor, W / down_scaling_factor)

        # Decoder to embedding space
        emb = self.decoder(
            fused
        )  # (B, embed_dim, H / down_scaling_factor, W / down_scaling_factor)

        # L2 normalize for metric learning
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


class DinoSegmentation(nn.Module):
    def __init__(
        self,
        dino_model,
        num_classes,
        dino_out_dim=384,
        cnn_out_dim=128,
        embed_dim=128,
    ):
        super().__init__()
        self.feature_extractor = DinoMetricLearning(
            dino_model, dino_out_dim, cnn_out_dim, embed_dim
        )
        self.segmentation_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 3, 1)
        x = self.segmentation_head(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DinoUpsampling(nn.Module):
    def __init__(self, dino_model, out_dim=512):
        super().__init__()
        self.dino = FrozenDINOv2(dino_model)

        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(dino_model.embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        B, _, H, W = x.shape

        # Compute DINO features and upsample to original resolution
        feats = self.dino(x)
        print(feats.shape)
        feats = F.interpolate(
            feats, size=(H, W), mode="bilinear", align_corners=False
        )  # (B, C, H, W)
        print(feats.shape)

        feats = feats.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Apply fully connected layers
        feats = self.fc(feats)

        feats = feats.permute(0, 3, 1, 2)  # (B, C, H, W)

        return feats
