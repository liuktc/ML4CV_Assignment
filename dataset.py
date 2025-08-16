import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad
import random


class StreetHazardDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        image_transform=None,
        target_transform=None,
        positive_pairs=False,
        pixel_per_class=6,
    ):
        # Read the JSON file
        with open(annotations_file, "r") as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.positive_pairs = positive_pairs
        self.pixel_per_class = pixel_per_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.img_dir, sample["fpath_img"])
        seg_path = os.path.join(self.img_dir, sample["fpath_segm"])

        image = Image.open(img_path).convert("RGB")
        segmentation = Image.open(seg_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.target_transform:
            segmentation = self.target_transform(segmentation)
            segmentation = (
                torch.as_tensor(np.array(segmentation)) - 1
            )  # -1 to have classes from 0 to 12

        if not self.positive_pairs:
            return image, segmentation.long()
        else:
            """
            For the metric learning task, we need to generate positive pairs (pairs of pixels that belong to the same class).
            We do that in a way that we take the same amount of positive pairs for each class in order to better train the model.
            """
            # Different classes in the segmentation
            classes = torch.unique(segmentation)
            # Remove class 255, which is the ignore class produced by the padding
            classes = classes[classes != 255]
            selected_pixels = []
            # Number of pixels to select in total
            N = 13 * self.pixel_per_class
            target_matrix = torch.zeros((N, N), dtype=torch.long)

            # For each class, try to take the same amount of positive pair
            selected_num_pixel_per_class = torch.tensor(
                [
                    min((segmentation == c).sum().item(), self.pixel_per_class)
                    for c in classes
                ]
            )
            num_pixel_per_class = torch.tensor(
                [(segmentation == c).sum().item() for c in classes]
            )

            # If a class dont have enough pixel, we need to take more pixel from other classes
            classes_with_not_enough_pixel = classes[
                selected_num_pixel_per_class < self.pixel_per_class
            ]
            num_pixels_remaining = N - selected_num_pixel_per_class.sum().item()

            # Assign the remaining pixels to the classes that have enough pixels
            while num_pixels_remaining > 0:
                for index, c in enumerate(classes):
                    if (
                        c not in classes_with_not_enough_pixel
                        and num_pixel_per_class[index]
                        > selected_num_pixel_per_class[index]
                    ):
                        selected_num_pixel_per_class[index] += 1
                        num_pixels_remaining -= 1
                        if num_pixels_remaining <= 0:
                            break

            # Build the target tensor
            tot = 0
            for n in selected_num_pixel_per_class:
                n = n.item()
                target_matrix[tot : tot + n, tot : tot + n] = 1
                tot += n

            selected_pixels = torch.tensor([])

            for index, c in enumerate(classes):
                flattened_segmentation = segmentation.flatten()
                mask = flattened_segmentation == c
                mask_len = mask.sum().item()
                mask_indexes = torch.arange(len(flattened_segmentation))[mask]
                # Select a random permutation of the indexes
                random_perm = torch.randperm(mask_len)[
                    : selected_num_pixel_per_class[index]
                ]
                selected_indexes = mask_indexes[random_perm]
                selected_pixels = torch.cat((selected_pixels, selected_indexes))

            return image, segmentation.long(), (selected_pixels.long(), target_matrix)


# class StreetHazardDatasetTriplet(Dataset)


class StreetHazardDatasetTriplet(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        image_transform=None,
        target_transform=None,
    ):
        # Read the JSON file
        with open(annotations_file, "r") as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.img_dir, sample["fpath_img"])
        seg_path = os.path.join(self.img_dir, sample["fpath_segm"])

        image = Image.open(img_path).convert("RGB")
        segmentation = Image.open(seg_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.target_transform:
            segmentation = self.target_transform(segmentation)
            segmentation = torch.as_tensor(np.array(segmentation)) - 1

        segmentation = segmentation.long()
        classes = torch.unique(segmentation)
        classes = classes[classes != 255]  # ignore padding class

        # Select a random anchor class
        anchor_class = random.choice(classes).item()
        mask_anchor = (segmentation == anchor_class).nonzero(as_tuple=False)
        anchor_idx = mask_anchor[torch.randint(len(mask_anchor), (1,))].squeeze(0)

        # Positive: pick another pixel from the same class
        positive_idx = anchor_idx
        while torch.equal(positive_idx, anchor_idx):
            positive_idx = mask_anchor[torch.randint(len(mask_anchor), (1,))].squeeze(0)

        # Negative: pick a pixel from a different class
        negative_class = random.choice(classes[classes != anchor_class]).item()
        mask_negative = (segmentation == negative_class).nonzero(as_tuple=False)
        negative_idx = mask_negative[torch.randint(len(mask_negative), (1,))].squeeze(0)

        return image, segmentation, (anchor_idx, positive_idx, negative_idx)


class PadToMultipleOf14:
    """
    Simple padding to make the image dimensions multiple of 14.
    That's because DinoV2 requires the image dimensions to be multiple of 14.
    """

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        # Get image dimensions (assumes input is a PIL image)
        width, height = image.size
        # Calculate the new dimensions
        new_width = ((width + 13) // 14) * 14
        new_height = ((height + 13) // 14) * 14
        # Calculate padding amounts
        pad_left = 0
        pad_top = 0
        pad_right = new_width - width
        pad_bottom = new_height - height
        # Apply padding
        return pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

    def convert_dims(self, size):
        width, height = size
        new_width = ((width + 13) // 14) * 14
        new_height = ((height + 13) // 14) * 14
        return new_width, new_height
