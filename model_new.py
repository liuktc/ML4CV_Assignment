import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from sklearn.covariance import LedoitWolf
from utils import sample_pixels_per_class
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

# -------------------------------
# 1. Frozen DINOv2 wrapper
# -------------------------------


class FrozenDINOv2(nn.Module):
    def __init__(self, dinov2_model, out_dim=None, layers_to_use=None, fusion="concat"):
        """
        Args:
            dinov2_model: pretrained DINOv2 backbone
            out_dim: output embedding dim after projection (None = no projection)
            layers_to_use: int or list of layer indices to extract (e.g. [3, 6, 9, 11])
            fusion: "concat" or "sum"
        """
        super().__init__()
        self.dino = dinov2_model
        for p in self.dino.parameters():
            p.requires_grad = False

        self.layers_to_use = layers_to_use or [len(self.dino.blocks) - 1]
        self.fusion = fusion

        embed_dim = self.dino.embed_dim
        if fusion == "concat":
            in_dim = embed_dim * len(self.layers_to_use)
        elif fusion == "sum":
            in_dim = embed_dim
        else:
            raise ValueError("fusion must be 'concat' or 'sum'")

        if out_dim is not None:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            fused features: (B, out_dim, H/patch, W/patch)
        """
        B, C, H, W = x.shape
        Hp, Wp = H // self.dino.patch_size, W // self.dino.patch_size

        # Call backbone correctly: (x, n, return_class_token=...)
        feats = self.dino.get_intermediate_layers(x, n=12)
        feats_per_layer = [feats[i] for i in self.layers_to_use]

        selected_feats = []
        for f in feats_per_layer:
            # f: (B, num_patches, C) → reshape to (B, C, Hp, Wp)
            f = f.view(B, Hp, Wp, -1)
            f = f.permute(0, 3, 1, 2).contiguous()
            selected_feats.append(f)

        # fuse
        if self.fusion == "concat":
            fused = torch.cat(selected_feats, dim=1)
        else:  # "sum"
            fused = torch.stack(selected_feats, dim=0).sum(dim=0)

        return self.proj(fused)  # (B, out_dim, Hp, Wp)


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
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, embed_dim, 1),  # projection to embedding dim
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
    def __init__(self, dino_model, out_dim=512, normalize: bool = True):
        super().__init__()
        self.dino = FrozenDINOv2(dino_model)

        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(dino_model.embed_dim, out_dim),
            # nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        self.normalize = normalize

    def forward(self, x):
        B, _, H, W = x.shape

        # Compute DINO features and upsample to original resolution
        feats = self.dino(x)
        feats = F.interpolate(
            feats, size=(H, W), mode="bilinear", align_corners=False
        )  # (B, C, H, W)

        feats = feats.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Apply fully connected layers
        feats = self.fc(feats)

        feats = feats.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Normalize features
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)

        return feats


def pixel_embeddings(backbone, x):
    # x: (B, C, H, W)
    feats = backbone(x)  # (B, d, H_p, W_p)
    B, d, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, d)  # (B*H*W, d)
    return feats, (H, W)


class OutlierDetector(nn.Module):
    def __init__(
        self,
        model,
        num_classes,
        train_dataloader,
        pca_dim=50,
        device="cpu",
        num_samples_per_class=50,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.device = device
        self.pca_dim = pca_dim
        self.model = model
        self.model.eval()

        # Setup
        self.class_pca = [
            IncrementalPCA(n_components=pca_dim, whiten=True)
            for _ in range(num_classes)
        ]
        class_cov = [LedoitWolf() for _ in range(num_classes)]

        # Temporary buffer to accumulate features per class
        accum_feats = defaultdict(list)

        # Step 1: fit PCA incrementally per class
        for x, labels, _ in tqdm(train_dataloader, desc="Fitting PCA per class"):
            x = x.to(device)  # (B,C,H,W)
            labels = labels.to(device)  # (B,H,W)
            feats, (H, W) = pixel_embeddings(self.model, x)  # (B*H*W, d)
            labels_flat = labels.reshape(-1)  # (B*H*W,)

            feats = feats.detach().cpu()
            labels_flat = labels_flat.detach().cpu()

            # Sample pixels per class
            feats, labels_flat = sample_pixels_per_class(
                feats, labels_flat, num_samples_per_class=num_samples_per_class
            )

            for c in range(num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue
                feats_c = feats[mask].numpy()

                # Add to buffer
                accum_feats[c].append(feats_c)

                # Concatenate if enough samples
                accumulated = np.concatenate(accum_feats[c], axis=0)
                if accumulated.shape[0] >= pca_dim:
                    # Only take the first 'accumulated.shape[0]' samples (can take all or slice)
                    self.class_pca[c].partial_fit(accumulated)
                    # Clear buffer after fitting
                    accum_feats[c] = []

        for c in range(num_classes):
            if len(accum_feats[c]) > 0:
                remaining = np.concatenate(accum_feats[c], axis=0)
                self.class_pca[c].partial_fit(remaining)

        feats_c_pca_list = []
        # Free gpu memory
        del feats, labels_flat
        torch.cuda.empty_cache()

        # Step 2: fit Gaussian per class in PCA space
        for x, labels, _ in tqdm(train_dataloader, desc="Fitting Gaussian per class"):
            x = x.to(device)
            labels = labels.to(device)
            feats, (H, W) = pixel_embeddings(self.model, x)
            labels_flat = labels.reshape(-1)

            feats = feats.detach().cpu()
            labels_flat = labels_flat.detach().cpu()

            for c in range(num_classes):
                mask = labels_flat == c
                if mask.sum() == 0:
                    continue
                feats_c = feats[mask].detach().cpu().numpy()
                feats_c_pca = self.class_pca[c].transform(feats_c)
                feats_c_pca_list.append(feats_c_pca)

        for c in range(num_classes):
            if len(feats_c_pca_list) == 0:
                continue
            feats_c_pca_all = np.vstack(feats_c_pca_list)
            class_cov[c].fit(feats_c_pca_all)

        # Store mean & inverse covariance per class
        self.mu_c = [
            torch.tensor(class_cov[c].location_, dtype=torch.float32)
            for c in range(num_classes)
        ]
        self.cov_inv_c = [
            torch.tensor(np.linalg.inv(class_cov[c].covariance_), dtype=torch.float32)
            for c in range(num_classes)
        ]

    # Step 3: Inference: per-pixel Mahalanobis map
    def forward(self, x):
        feats, (H, W) = pixel_embeddings(self.model, x.to(self.device))  # (HW, d)
        dists_per_class = []
        feats = feats.detach().cpu()

        for c in range(self.num_classes):
            # if mu_c[c] is None or cov_inv_c[c] is None:
            try:
                feats_pca = torch.tensor(
                    self.class_pca[c].transform(feats.detach().cpu().numpy()),
                    dtype=torch.float32,
                )
                diff = feats_pca - self.mu_c[c]
                dists = (diff @ self.cov_inv_c[c] * diff).sum(-1)  # (HW,)
                dists_per_class.append(dists.unsqueeze(1))
            except AttributeError:
                dists_per_class.append(torch.full((H * W, 1), float("inf")))

        dists_all = torch.cat(dists_per_class, dim=1)  # (HW, num_classes)
        min_dists, _ = torch.min(dists_all, dim=1)  # closest class distance
        return min_dists.reshape(H, W)  # anomaly map


class ProxyOutlierDetector(nn.Module):
    def __init__(self, model, proxies, device="cpu", normalize=True):
        """
        Args:
            model: backbone feature extractor (frozen DINOv3 for example)
            proxies: (num_classes, d) tensor of learned ProxyAnchor proxies
            device: device
            normalize: whether to L2 normalize proxies & embeddings before scoring
        """
        super().__init__()
        self.model = model.eval()
        self.device = device
        self.normalize = normalize

        # Store proxies
        self.register_buffer("proxies", proxies.clone().to(device))

        if self.normalize:
            self.proxies = F.normalize(self.proxies, p=2, dim=1)

    def forward(self, x):
        """
        Args:
            x: input image tensor (B,C,H,W)
        Returns:
            anomaly maps per image (B, H_p, W_p)
        """
        feats, (H, W) = pixel_embeddings(self.model, x.to(self.device))  # (B*H*W, d)
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)

        # Compute Euclidean distance from each proxy
        # feats: (N, d), proxies: (num_classes, d)
        # dist = sqrt(sum((x - p)^2))
        dists = torch.cdist(feats, self.proxies, p=2)  # (B*H*W, num_classes)

        # For anomaly detection, take min distance (distance from closest proxy)
        min_dist, _ = torch.min(dists, dim=1)  # (B*H*W,)
        anomaly_score = min_dist

        # Reshape back to spatial map
        B = x.shape[0]
        anomaly_score = anomaly_score.view(B, H, W)

        return anomaly_score
