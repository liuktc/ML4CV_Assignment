import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from plot import plot_semantic_segmentation, semantic_embeddings_plot, color
from torch.utils.tensorboard import SummaryWriter
from dataset import sample_pixels_per_class
from model_new import OutlierDetector, EnergyBasedOutlierDetector
from metrics import outlier_detection_roc_auc, compute_aupr, compute_mIoU
import cv2


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path
        self.best_state_dict = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
                self.best_state_dict = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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
    metric_learning_loss: nn.Module,
    segmentation_loss: nn.Module,
    loss_weighting: nn.Module,
    dl_train: DataLoader,
    dl_train_small: DataLoader,  # Small subset of training data for metrics
    test_dataset: DataLoader,  # Dataset for computing test metrics !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    mining_func=None,
    plot_interval: int = 1,
    log_dir: str = "./runs",  # TensorBoard log directory
    print_loss: bool = False,
    num_classes: int = 13,
    save_path: str = "metric_learning_model.pth",  # Path to save the best model
    pixel_per_class: int = 50,  # Number of pixels to sample per class
):
    writer = SummaryWriter(log_dir)
    global_step = 0
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path=save_path)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, segmentations) in tqdm(
            enumerate(dl_train), total=len(dl_train)
        ):
            images = images.to(device)
            segmentations = segmentations.to(device)

            logits, embeddings = model(images, return_features=True)

            B, D, H, W = embeddings.shape
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(B * H * W, D)
            # Reshape segmentations to (B*H*W)
            labels = segmentations.view(B * H * W)

            embeddings, labels = sample_pixels_per_class(
                embeddings, labels, num_samples_per_class=pixel_per_class
            )

            if mining_func is None:
                indices_tuple = None
            else:
                indices_tuple = mining_func(embeddings, labels)

            # Compute losses
            metric_learning_loss_value = metric_learning_loss(
                embeddings, labels, indices_tuple
            )
            segmentation_loss_value = segmentation_loss(logits, segmentations)
            loss = loss_weighting(metric_learning_loss_value, segmentation_loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar(
                "Train/Metric_Learning_Loss",
                metric_learning_loss_value.item(),
                global_step,
            )
            writer.add_scalar(
                "Train/Segmentation_Loss", segmentation_loss_value.item(), global_step
            )

            if print_loss:
                print(
                    f"Epoch {epoch}/{epochs} - Step {global_step} - Loss: {loss.item()}"
                )
            global_step += 1

            if i % plot_interval == 0:
                with torch.no_grad():
                    print(f"Epoch {epoch}/{epochs} - {loss.item()}")
                    image = semantic_embeddings_plot(
                        model, dl_train, num_points=3000, device=device
                    )
                    writer.add_image(
                        "Train/Semantic Embeddings",
                        image,
                        global_step=global_step,
                    )
                    avg_train_loss = running_train_loss / (i + 1)
                    early_stopping(avg_train_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        writer.close()
                        return

                    # if i % metric_interval == 0 and i > 0:
                    # Compute test metrics
                    detector = EnergyBasedOutlierDetector(
                        model, temperature=1, device=device
                    )
                    # detector = OutlierDetector(
                    #     model,
                    #     num_classes=num_classes,
                    #     train_dataloader=dl_train_small,
                    #     device=device,
                    #     num_samples_per_class=512,
                    #     pca_dim=128,
                    # )

                    # Compute the metric over 10 random test images
                    scores = {"AUPR": [], "mIoU": []}
                    test_indices = torch.randperm(len(test_dataset))[:2]
                    rows = []
                    for idx in tqdm(test_indices, desc="Computing test metrics"):
                        test_image, test_segmentation = test_dataset[idx]
                        # Print unique values in test_segmentation
                        outliers_map = detector(test_image.unsqueeze(0).to(device))
                        outliers_gt = (test_segmentation == 13).int()

                        test_image = test_image.to(device).unsqueeze(0)
                        test_logits = model(test_image)
                        predicted_segmentation = (
                            torch.argmax(test_logits, dim=1).squeeze(0).detach().cpu()
                        )

                        unnormalized_image = test_image * torch.tensor(
                            [0.229, 0.224, 0.225]
                        ).view(1, 3, 1, 1).to(device) + torch.tensor(
                            [0.485, 0.456, 0.406]
                        ).view(1, 3, 1, 1).to(device)

                        # Create a heatmap image from the outliers_map using a colormap
                        outliers_map_img = (
                            outliers_map.squeeze(0).squeeze(0).detach().cpu().numpy()
                        )
                        outliers_map_img = (outliers_map_img * 255).astype("uint8")
                        outliers_map_img = cv2.applyColorMap(
                            outliers_map_img, cv2.COLORMAP_MAGMA
                        )
                        outliers_map_img = torch.tensor(outliers_map_img).permute(
                            2, 0, 1
                        )
                        outliers_map_img = outliers_map_img.to(device)
                        outliers_map_img = outliers_map_img.float() / 255.0

                        rows.append(
                            torch.cat(
                                [
                                    unnormalized_image[0],
                                    torch.Tensor(
                                        color(test_segmentation, return_array=True)
                                    )
                                    .to(device)
                                    .permute(2, 0, 1),
                                    torch.Tensor(
                                        color(predicted_segmentation, return_array=True)
                                    )
                                    .to(device)
                                    .permute(2, 0, 1),
                                    outliers_map_img,
                                ],
                                dim=-1,
                            )
                        )

                        aupr = compute_aupr(
                            outliers_gt.detach().cpu(), outliers_map.detach().cpu()
                        )
                        miou = compute_mIoU(
                            test_segmentation.detach().cpu(),
                            predicted_segmentation.detach().cpu(),
                            num_classes + 1,
                        )

                        scores["AUPR"].append(aupr)
                        scores["mIoU"].append(miou)

                    grid = torch.cat(rows, dim=-2)  # stack rows vertically
                    grid = grid.detach()
                    # Reduce the grid size by 4
                    grid = torch.nn.functional.interpolate(
                        grid.unsqueeze(0), scale_factor=0.4, mode="bilinear"
                    ).squeeze(0)
                    writer.add_image("Test/Results", grid, global_step=global_step)

                    for key in scores:
                        mean_score = sum(scores[key]) / len(scores[key])
                        std_score = torch.std(torch.tensor(scores[key]))
                        writer.add_scalar(f"Test/{key}/mean", mean_score, global_step)
                        writer.add_scalar(
                            f"Test/{key}/lower", mean_score - std_score, global_step
                        )
                        writer.add_scalar(
                            f"Test/{key}/upper", mean_score + std_score, global_step
                        )

                        print(f"{key} mean: {mean_score} ± {std_score}")

        avg_train_loss = running_train_loss / len(dl_train)
