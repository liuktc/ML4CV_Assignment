from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image

COLORS = np.array(
    [
        [0, 0, 0],  # unlabeled    =   0,
        [70, 70, 70],  # building     =   1,
        [190, 153, 153],  # fence        =   2,
        [250, 170, 160],  # other        =   3,
        [220, 20, 60],  # pedestrian   =   4,
        [153, 153, 153],  # pole         =   5,
        [157, 234, 50],  # road line    =   6,
        [128, 64, 128],  # road         =   7,
        [244, 35, 232],  # sidewalk     =   8,
        [107, 142, 35],  # vegetation   =   9,
        [0, 0, 142],  # car          =  10,
        [102, 102, 156],  # wall         =  11,
        [220, 220, 0],  # traffic sign =  12,
        [60, 250, 240],  # anomaly      =  13,
    ]
)

LABEL_NAMES = [
    "unlabeled",
    "building",
    "fence",
    "other",
    "pedestrian",
    "pole",
    "road line",
    "road",
    "sidewalk",
    "vegetation",
    "car",
    "wall",
    "traffic sign",
    "anomaly",
]


def color(annotated_image: np.ndarray, colors: np.ndarray = COLORS) -> Image.Image:
    img_new = np.zeros((*annotated_image.shape, 3), dtype=np.uint8)

    for index, color in enumerate(colors):
        img_new[annotated_image == index] = color

    return Image.fromarray(img_new, "RGB")


def pad_to_multiple_of_14(image):
    _, h, w = image.shape  # Assume image shape is (C, H, W)
    new_h = ((h + 13) // 14) * 14
    new_w = ((w + 13) // 14) * 14
    pad_h = new_h - h
    pad_w = new_w - w
    # Padding format: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode="constant", value=0)  # Zero padding
    return padded_image


def semantic_embeddings_plot(model, dl, num_points: int = 100, device: str = "cpu"):
    for images, segmentations, _ in dl:
        image = images[0].to(device)
        segmentation = segmentations[0].to(device)
        plt.imshow(image.permute(1, 2, 0).cpu())
        plt.show()
        plt.imshow(color(segmentation.cpu()))
        plt.show()

        out = model(image[None, ...])
        # Apply PCA on the channel dimension of the output embeddings
        out = out[0]
        pca = PCA(n_components=2)
        C, H, W = out.shape
        out = torch.permute(out, (1, 2, 0))
        out = out.view(-1, out.size(-1))
        print(out.shape)
        pca_out = pca.fit_transform(out.cpu().detach().numpy())
        pca_out = pca_out.reshape(H, W, 2)
        print(pca_out.shape)

        # Pick 1000 random pixels
        selected_pixels_x = torch.randperm(H)[:num_points]
        selected_pixels_y = torch.randperm(W)[:num_points]
        different_segmentation_values = torch.unique(segmentation)

        print(different_segmentation_values)

        for i in range(len(different_segmentation_values)):
            plt.scatter(
                x=pca_out[selected_pixels_x, selected_pixels_y][
                    segmentation[selected_pixels_x, selected_pixels_y]
                    == different_segmentation_values[i].item(),
                    0,
                ],
                y=pca_out[selected_pixels_x, selected_pixels_y][
                    segmentation[selected_pixels_x, selected_pixels_y]
                    == different_segmentation_values[i].item(),
                    1,
                ],
                label=different_segmentation_values[i].item(),
            )
        plt.legend()
        plt.show()
        break


def plot_semantic_segmentation(image, predicted, gt):
    plt.subplots(1, 3, figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(color(predicted.argmax(dim=0).cpu()))
    plt.title("Predicted Segmentation")
    plt.subplot(1, 3, 3)
    plt.imshow(color(gt.cpu()))
    plt.title("Ground Truth Segmentation")
    plt.show()
