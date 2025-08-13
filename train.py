import torch
import torch.nn as nn
from tqdm.auto import tqdm
from _model import DINOv2_SemanticSegmentation
from torch.utils.data import DataLoader
from plot import plot_semantic_segmentation
from torch.utils.tensorboard import SummaryWriter


def train_semantic_segmentation(
    model: DINOv2_SemanticSegmentation,
    criterion: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,  # Add validation DataLoader
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    original_image_size: tuple = (720, 1280),
    plot_interval: int = 1,
    log_dir: str = "./runs",  # TensorBoard log directory
):
    writer = SummaryWriter(log_dir)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, segmentations, _) in tqdm(enumerate(dl_train)):
            images = images.to(device)
            segmentations = segmentations.to(device)

            logits = model(images)

            # Crop segmentations and logits to the original size
            segmentations = segmentations[
                :, : original_image_size[0], : original_image_size[1]
            ]
            logits = logits[:, :, : original_image_size[0], : original_image_size[1]]

            loss = criterion(logits, segmentations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - {loss.item()}")
                plot_semantic_segmentation(images[0], logits[0], segmentations[0])

        avg_train_loss = running_train_loss / len(dl_train)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, segmentations, _ in dl_val:
                images = images.to(device)
                segmentations = segmentations.to(device)

                logits = model(images)
                segmentations = segmentations[
                    :, : original_image_size[0], : original_image_size[1]
                ]
                logits = logits[
                    :, :, : original_image_size[0], : original_image_size[1]
                ]

                val_loss = criterion(logits, segmentations)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(dl_val)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if (epoch + 1) % plot_interval == 0:
            plot_semantic_segmentation(images[0], logits[0], segmentations[0])

    writer.close()
