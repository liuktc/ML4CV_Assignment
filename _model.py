import torch
import torch.nn as nn
from typing import Tuple


class DINOv2_SemanticSegmentation(nn.Module):
    def __init__(
        self, num_classes: int, image_size: Tuple[int, int], embedding_size: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes

        self.feature_extractor = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )

        self.head = SegmentationHead(
            num_classes,
            d_model=384,
            image_size=image_size,
            embedding_size=embedding_size,
        )

        self.fine_grained_features_extractor = nn.Sequential(
            nn.Conv2d(3, embedding_size // 2, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(embedding_size // 2),
            nn.Conv2d(
                embedding_size // 2,
                embedding_size,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(embedding_size),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        H = H // 14
        W = W // 14
        y = self.feature_extractor(x, is_training=True)
        backbone_features = y["x_norm_patchtokens"]

        B, HW, d_model = backbone_features.shape
        backbone_features = backbone_features.view((B, H, W, d_model))
        backbone_features = torch.permute(backbone_features, (0, 3, 1, 2))

        fine_grained_features = self.fine_grained_features_extractor(x)

        return self.head(backbone_features, fine_grained_features)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int,
        image_size: Tuple[int, int],
        embedding_size: int = 256,
    ):
        """Segmentation Head for DINOv2

        Args:
            num_classes (int): The number of classes to segment (if None, it will output all the features)
            d_model (int): The dimension of the input embeddings from the backbone
            image_size (Tuple[int, int]): The size of the output image
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        self.l1 = nn.Linear(d_model, embedding_size)

        self.up = nn.Upsample(image_size)
        if num_classes is not None:
            # Segmentation
            self.out = nn.Conv2d(
                embedding_size,
                num_classes,
                kernel_size=1,
                stride=1,
                padding="same",
                padding_mode="reflect",
            )
        else:
            # Metric Learning
            self.out = nn.Conv2d(
                embedding_size,
                embedding_size,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode="reflect",
            )

        # self.out = nn.Sequential(
        #     nn.Conv2d(512, 128, kernel_size=3, stride=1, padding="same"),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding="same"),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding="same"),
        # )

    def forward(self, backbone_features, fine_grained_features):
        """
        First the backbone features are upsampled to match the image size.
        Then the fine-grained features are concatenated to the backbone features.
        Finally, the output is computed with a Conv2D.
        """
        B, d_model, H, W = backbone_features.shape
        backbone_features = torch.permute(backbone_features, (0, 2, 3, 1))
        # (B, H, W, d_model)

        backbone_features = self.l1(backbone_features)
        # (B, H, W, embedding_size)

        backbone_features = torch.permute(backbone_features, (0, 3, 1, 2))
        # (B, embedding_size, H, W)

        backbone_features = self.up(backbone_features)
        # (B, embedding_size, image_size, image_size)

        # Sum fine-grained features with backbone features
        features = backbone_features + fine_grained_features
        # (B, embedding_size, image_size, image_size)

        out = self.out(features)
        # (B, num_classes or embedding_size, image_size, image_size)
        return out
