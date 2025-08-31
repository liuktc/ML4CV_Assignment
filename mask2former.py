# mask2former_pytorch.py
# Pure PyTorch "Mask2Former-like" implementation (simplified)
# Requirements: torch, torchvision (for backbone if desired)
# Works CPU/GPU.

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Utilities
# -------------------------------
def build_norm(norm_type: str, dim: int):
    if norm_type == "BN":
        return nn.BatchNorm2d(dim)
    elif norm_type == "GN":
        return nn.GroupNorm(32, dim) if dim % 32 == 0 else nn.GroupNorm(8, dim)
    elif norm_type == "LN":
        # LayerNorm works on channels-last or 1D; we wrap a small module
        return nn.GroupNorm(1, dim)  # approximate LN for conv features
    else:
        raise ValueError(norm_type)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Simple Backbone (ResNet-18) adapter
# (You can plug any backbone that produces multi-scale feature maps.)
# -------------------------------
from torchvision.models import resnet18


class SimpleResNetBackbone(nn.Module):
    """
    Returns feature maps at strides [4,8,16,32] similar to typical FPN.
    """

    def __init__(self, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        res = resnet18(pretrained=pretrained)
        # Keep layers up to layer4
        self.conv1 = res.conv1
        self.bn1 = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.layer1 = res.layer1  # stride 4 output (after conv1+pool)
        self.layer2 = res.layer2  # stride 8
        self.layer3 = res.layer3  # stride 16
        self.layer4 = res.layer4  # stride 32

        # project to out_channels
        self.outc2 = nn.Conv2d(64, out_channels, 1)
        self.outc3 = nn.Conv2d(128, out_channels, 1)
        self.outc4 = nn.Conv2d(256, out_channels, 1)
        self.outc5 = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4
        c2 = self.layer1(x)  # /4 (approx)
        c3 = self.layer2(c2)  # /8
        c4 = self.layer3(c3)  # /16
        c5 = self.layer4(c4)  # /32

        p2 = self.outc2(c2)
        p3 = self.outc3(c3)
        p4 = self.outc4(c4)
        p5 = self.outc5(c5)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


# -------------------------------
# Pixel Decoder (FPN-like)
# -------------------------------
class FPNPixelDecoder(nn.Module):
    """
    Simple FPN that upsamples lower-res features and fuses them.
    Produces a single unified feature map at the highest resolution (p2 stride).
    Also returns intermediate pyramid features for the transformer cross-attention.
    """

    def __init__(self, in_channels=256, out_channels=256, norm="GN"):
        super().__init__()
        self.lateral5 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels, out_channels, 1)

        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.out_channels = out_channels
        self.norm = norm

        # optional conv to produce mask feature
        self.mask_feature = nn.Conv2d(out_channels, out_channels, 1)

    def forward(
        self, feats: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        feats: dict with keys p2, p3, p4, p5 (p2 highest res)
        returns:
            pyramid: {"p2":..., "p3":..., "p4":..., "p5":...}
            fused: unified high-res feature [B, out_channels, H2, W2] (same as p2)
        """
        p2 = feats["p2"]
        p3 = feats["p3"]
        p4 = feats["p4"]
        p5 = feats["p5"]

        # lateral
        l5 = self.lateral5(p5)
        l4 = self.lateral4(p4)
        l3 = self.lateral3(p3)
        l2 = self.lateral2(p2)

        # top-down
        up5 = F.interpolate(
            l5, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4_t = self.smooth4(l4 + up5)

        up4 = F.interpolate(
            p4_t, size=l3.shape[-2:], mode="bilinear", align_corners=False
        )
        p3_t = self.smooth3(l3 + up4)

        up3 = F.interpolate(
            p3_t, size=l2.shape[-2:], mode="bilinear", align_corners=False
        )
        p2_t = self.smooth2(l2 + up3)

        pyramid = {"p2": p2_t, "p3": p3_t, "p4": p4_t, "p5": l5}
        fused = self.mask_feature(p2_t)  # high-res feature for mask prediction
        return pyramid, fused


# -------------------------------
# Positional encoding for 2D maps (sine cosine)
# -------------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.linspace(0, 1, steps=H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.linspace(0, 1, steps=W, device=device).unsqueeze(0).repeat(H, 1)
        # create channels
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed.unsqueeze(0) / dim_t.view(-1, 1, 1)
        pos_y = y_embed.unsqueeze(0) / dim_t.view(-1, 1, 1)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=1).flatten(1, 2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=1).flatten(1, 2)
        pos = torch.cat((pos_y, pos_x), dim=1)  # [num_pos_feats*2, H, W]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)
        return pos[:, : self.num_pos_feats * 2, :, :]


# -------------------------------
# Multi-Head Attention with projection convenience
# -------------------------------
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, q, k, v, key_padding_mask=None):
        # q: [B, Lq, C], k/v: [B, Lk, C]
        out, attn = self.attn(
            q, k, v, key_padding_mask=key_padding_mask, need_weights=True
        )
        return out, attn


# -------------------------------
# Transformer Decoder (simplified)
# -------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, memory_pos=None):
        # tgt: [B, nq, C], memory: [B, N, C]
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross-attention
        # optionally add memory_pos to memory embeddings
        if memory_pos is not None:
            mem = memory + memory_pos
        else:
            mem = memory
        tgt2, attn = self.multihead_attn(tgt, mem, mem)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, memory_pos=None):
        attns = []
        for layer in self.layers:
            tgt, attn = layer(tgt, memory, memory_pos=memory_pos)
            attns.append(attn)
        return tgt, attns


# -------------------------------
# Mask2Former Head (queries -> classes + masks)
# -------------------------------
class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_classes: int = 21,
        num_queries: int = 100,
        feature_size: int = 256,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        # transformer decoder
        dec_layer = TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4
        )
        self.decoder = TransformerDecoder(dec_layer, num_layers=6)

        # outputs
        self.class_head = nn.Linear(
            hidden_dim, num_classes + 1
        )  # +1 for "no object"/background
        # mask embedding projection: produce per-query mask embedding maps that dot with pixel features to form masks
        self.mask_embed_head = MLP(hidden_dim, hidden_dim, feature_size, n_layers=3)

    def forward(
        self,
        pixel_feats: torch.Tensor,
        pixel_pos: torch.Tensor,
        pyramid_feats: Dict[str, torch.Tensor],
    ):
        """
        pixel_feats: fused high-res feature [B, C, H, W] (from FPNPixelDecoder)
        pixel_pos: positional encoding same spatial size as pixel_feats [B, Cpos, H, W] (Cpos maybe 256)
        pyramid_feats: dict of multi-scale features (not used heavily here but could be)
        """
        batch_size = pixel_feats.shape[0]
        B = batch_size
        # flatten memory for cross-attention: [B, N, C]
        C = pixel_feats.shape[1]
        H, W = pixel_feats.shape[2], pixel_feats.shape[3]
        memory = pixel_feats.flatten(2).permute(0, 2, 1).contiguous()  # [B, N, C]
        # positional encoding flatten
        if pixel_pos is not None:
            pos = pixel_pos.flatten(2).permute(0, 2, 1).contiguous()  # [B,N,Cpos]
        else:
            pos = None

        # initialize queries
        q = self.query_embed.weight.unsqueeze(0).repeat(
            B, 1, 1
        )  # [B, num_queries, hidden_dim]
        # Optionally add learned query positions
        q = q + self.query_pos.unsqueeze(0)

        # decoder
        hs, attns = self.decoder(q, memory, memory_pos=pos)  # hs: [B, nq, hidden_dim]

        # classification
        class_logits = self.class_head(hs)  # [B, nq, num_classes+1]

        # mask embeddings (per query): produce map embeddings per query (dot with pixel features to get mask)
        # mask_embed_head maps each query vector to a mask embedding vector of length feature_size
        mask_embeds = self.mask_embed_head(hs)  # [B, nq, feature_size]

        # produce mask logits by dot-product between mask_embeds and pixel features at each pixel:
        # pixel_feats: [B, Cpix, H, W] -> project to feature_size first if Cpix != feature_size
        # Here we assume pixel_feats channels == feature_size
        pf = pixel_feats  # [B, F, H, W]
        pf_flat = pf.view(B, pf.shape[1], -1)  # [B, F, H*W]
        # normalize or not
        # compute dot: for each query q: mask_logits_q = mask_embed_q (F) dot pf at each pixel => [B, nq, H*W]
        # we do batch matrix mul:
        # mask_embeds: [B, nq, F], pf_flat: [B, F, H*W] -> (B, nq, H*W)
        mask_logits = torch.einsum(
            "bqf, bfh -> bqh", mask_embeds, pf_flat
        )  # [B, nq, H*W]
        mask_logits = mask_logits.view(B, self.num_queries, H, W)
        return class_logits, mask_logits, attns


# -------------------------------
# Mask2Former full model
# -------------------------------
class Mask2Former(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 21,
        hidden_dim: int = 256,
        num_queries: int = 100,
        feature_size: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = FPNPixelDecoder(
            in_channels=feature_size, out_channels=feature_size
        )
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=feature_size)
        self.head = Mask2FormerHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            feature_size=feature_size,
        )
        # small projector if backbone channels != feature_size
        # assume backbone outputs channels == feature_size (our SimpleResNetBackbone does this)
        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, images: torch.Tensor, return_intermediate: bool = False):
        """
        images: [B,3,H,W] float normalized
        returns:
            logits: semantic logits per pixel. We'll combine query masks with class logits via weighted combination.
            masks: raw per-query mask logits [B, nq, H, W]
            class_logits: [B, nq, C+1]
        """
        feats = self.backbone(images)  # dict p2,p3,p4,p5
        pyramid, fused = self.pixel_decoder(
            feats
        )  # pyramid: dict, fused: [B, F, H2, W2]
        # positional embedding for fused
        pos = self.pos_embed(fused)  # [B, Cpos, H2, W2]

        class_logits, mask_logits, attns = self.head(
            fused, pos, pyramid
        )  # [B,nq,C+1], [B,nq,H2,W2]

        # combine per-query masks into per-pixel class logits:
        # Option A (simple): take for each query its predicted class (argmax), weight mask by class prob → per-class map.
        # Simpler: compute per-pixel class logits by multiplying mask logits by class scores and summing over queries.
        # class_logits_raw: [B, nq, C+1] -> probabilities over (C+1)
        class_prob = F.softmax(class_logits, dim=-1)  # [B,nq,C+1]
        # take only first C classes (exclude "no object" at index -1 or last)
        C = class_prob.shape[-1] - 1
        cls_scores = class_prob[..., :C]  # [B,nq,C]
        # mask_logits: [B,nq,H,W], apply sigmoid to masks to get per-query mask prob
        mask_prob = torch.sigmoid(mask_logits)  # [B,nq,H,W]

        # compute per-class per-pixel score: sum_q cls_score_qc * mask_prob_qxy
        # rearrange: (B, nq, C) * (B, nq, H, W) -> (B, C, H, W) via einsum
        # first, move channels
        per_class_map = torch.einsum(
            "bqc, bqhw -> bchw", cls_scores, mask_prob
        )  # [B, C, H, W]

        # optionally apply softmax across classes to get probabilities
        # per_pixel_logits = per_class_map  # already class scores
        return {
            "per_class_logits": per_class_map,
            "mask_logits": mask_logits,
            "class_logits": class_logits,
        }

    def get_embeddings(self, images: torch.Tensor):
        # convenience: return fused pixel features for metric learning
        feats = self.backbone(images)
        _, fused = self.pixel_decoder(feats)
        return fused  # [B, F, H, W]


# -------------------------------
# Losses (simple)
# -------------------------------
def dice_loss(pred_masks, target_masks, eps=1e-6):
    # pred_masks sigmoid probs [B, nq, H, W], target_masks {0,1} [B, nq, H, W]
    num = 2 * (pred_masks * target_masks).sum(dim=(-2, -1))
    den = pred_masks.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) + eps
    loss = 1 - (num / den)
    return loss.mean()


class Mask2FormerLoss(nn.Module):
    """
    Simple loss: matching is not implemented — we assume a fixed-order matching between queries and targets,
    which makes this simpler but less robust. For production, implement Hungarian matching between predicted queries and GT masks.
    """

    def __init__(self, num_classes, no_object_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.num_classes = num_classes
        self.no_object_weight = no_object_weight

    def forward(self, outputs, targets):
        """
        outputs: dict from model forward
        targets: list of dicts per image: {"labels": [K_i], "masks": [K_i, H, W]} where K_i is variable
        For simplicity we'll construct target tensors with fixed shape:
           target_class_matrix: [B, nq] class indices 0..C-1 or C for "no object"
           target_mask_matrix: [B, nq, H, W] binary masks matching assignment (we assume 1-to-1 ordering)
        WARNING: this simple variant expects targets aligned to queries ordering.
        """
        per_class_logits = outputs["per_class_logits"]  # [B, C, H, W]
        B, C, H, W = per_class_logits.shape

        # Basic segmentation loss: cross-entropy per pixel
        # Build pixel-level target by stacking GT masks into per-pixel class id
        pixel_target = torch.full(
            (B, H, W), fill_value=255, dtype=torch.long, device=per_class_logits.device
        )
        for b in range(B):
            t = targets[b]
            # t: dict with keys "labels" (K,), "masks" (K,H_img,W_img) already resized to H,W
            if len(t["labels"]) == 0:
                continue
            for k, lab in enumerate(t["labels"]):
                mask_k = t["masks"][k].to(per_class_logits.device)
                pixel_target[b][mask_k > 0.5] = lab

        # segmentation CE
        seg_loss = self.ce(per_class_logits, pixel_target)

        # auxiliary mask + dice losses (optional)
        # compute mask predictions directly from outputs (we have mask_logits)
        mask_logits = outputs["mask_logits"]  # [B, nq, H, W]
        # For mask losses we need target assignment per query — we'll create naive assignment: for query i, use target i if exists
        # Build target masks padded/truncated to nq
        target_mask_matrix = torch.zeros_like(mask_logits)
        target_class_matrix = torch.full(
            (B, mask_logits.shape[1]),
            fill_value=C,
            dtype=torch.long,
            device=mask_logits.device,
        )
        for b in range(B):
            t = targets[b]
            K = len(t["labels"])
            for k in range(min(K, mask_logits.shape[1])):
                target_mask_matrix[b, k] = t["masks"][k].to(mask_logits.device)
                target_class_matrix[b, k] = t["labels"][k]
        # classification loss per query
        class_logits = outputs["class_logits"]  # [B, nq, C+1]
        class_logits_flat = class_logits.view(B * class_logits.shape[1], -1)
        class_target_flat = target_class_matrix.view(B * target_class_matrix.shape[1])
        class_loss = F.cross_entropy(
            class_logits_flat, class_target_flat, ignore_index=C
        )

        # dice between predicted masks and target masks
        pred_masks = torch.sigmoid(mask_logits)
        dloss = dice_loss(pred_masks, target_mask_matrix)

        loss = seg_loss + class_loss + dloss
        return {
            "loss": loss,
            "seg_loss": seg_loss.detach(),
            "class_loss": class_loss.detach(),
            "dice_loss": dloss.detach(),
        }


# -------------------------------
# Minimal training / usage example with random data
# -------------------------------
def _test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # small backbone
    backbone = SimpleResNetBackbone(pretrained=False, out_channels=256).to(device)
    model = Mask2Former(
        backbone=backbone,
        num_classes=13,
        hidden_dim=256,
        num_queries=50,
        feature_size=256,
    ).to(device)

    # fake batch
    B = 2
    H = 128
    W = 128
    images = torch.randn(B, 3, H, W, device=device)

    outputs = model(images)
    print("per_class_logits.shape:", outputs["per_class_logits"].shape)
    print("mask_logits.shape:", outputs["mask_logits"].shape)
    print("class_logits.shape:", outputs["class_logits"].shape)

    # build fake targets aligned with queries for loss demo
    targets = []
    for b in range(B):
        # create 3 random binary masks
        K = 3
        labels = [torch.randint(0, 13, (1,)).item() for _ in range(K)]
        masks = []
        for k in range(K):
            m = torch.zeros(H // 1, W // 1)
            # random rectangle
            y0 = torch.randint(0, H // 2, (1,)).item()
            x0 = torch.randint(0, W // 2, (1,)).item()
            y1 = min(H - 1, y0 + torch.randint(4, H // 2, (1,)).item())
            x1 = min(W - 1, x0 + torch.randint(4, W // 2, (1,)).item())
            m[y0:y1, x0:x1] = 1.0
            masks.append(m)
        targets.append({"labels": labels, "masks": masks})

    criterion = Mask2FormerLoss(num_classes=13)
    loss_dict = criterion(outputs, targets)
    print(
        "loss:",
        loss_dict["loss"].item(),
        "components:",
        loss_dict["seg_loss"].item(),
        loss_dict["class_loss"].item(),
        loss_dict["dice_loss"].item(),
    )


if __name__ == "__main__":
    _test_forward()
