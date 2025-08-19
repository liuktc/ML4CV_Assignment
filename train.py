import torch
import torch.nn as nn
from tqdm.auto import tqdm
from _model import DINOv2_SemanticSegmentation
from model_new import DinoMetricLearning
from torch.utils.data import DataLoader
from plot import plot_semantic_segmentation, semantic_embeddings_plot
from torch.utils.tensorboard import SummaryWriter


def train_semantic_segmentation(
    model: nn.Module,
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

            if i % plot_interval == 0:
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

    writer.close()


def train_metric_learning(
    model: nn.Module,
    criterion: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,  # Add validation DataLoader
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    plot_interval: int = 1,
    log_dir: str = "./runs",  # TensorBoard log directory
    print_loss: bool = False,
):
    writer = SummaryWriter(log_dir)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, segmentations, (selected_pixels, target_matrix)) in tqdm(
            enumerate(dl_train)
        ):
            images = images.to(device)
            segmentations = segmentations.to(device)
            selected_pixels = selected_pixels.to(device)
            target_matrix = target_matrix.to(device)

            embeddings = model(images)

            loss = criterion(
                X=embeddings,
                labels=segmentations,
                selected_pixels=selected_pixels,
                target_matrix=target_matrix,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            if print_loss:
                print(
                    f"Epoch {epoch}/{epochs} - Step {global_step} - Loss: {loss.item()}"
                )
            global_step += 1

            if i % plot_interval == 0:
                print(f"Epoch {epoch}/{epochs} - {loss.item()}")
                semantic_embeddings_plot(
                    model, dl_train, num_points=1000, device=device
                )

        avg_train_loss = running_train_loss / len(dl_train)
        writer.add_scalar("Train/Avg_Loss", avg_train_loss, epoch)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, segmentations, (selected_pixels, target_matrix) in dl_val:
                images = images.to(device)
                segmentations = segmentations.to(device)
                selected_pixels = selected_pixels.to(device)
                target_matrix = target_matrix.to(device)

                embeddings = model(images)
                loss = criterion(embeddings, selected_pixels, target_matrix)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(dl_val)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )
