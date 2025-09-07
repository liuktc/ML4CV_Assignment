import torch
import torch.nn as nn
from torch.utils.data import Dataset
from metrics import compute_aupr, compute_mIoU
from tqdm.auto import tqdm


def compute_metrics(model: nn.Module, detector, test_dataset: Dataset, device: str):
    model.eval()

    metrics = {"aupr": [], "miou": []}

    for idx in tqdm(range(len(test_dataset))):
        test_image, test_segmentation = test_dataset[idx]

        outliers_map = detector(test_image.unsqueeze(0).to(device))
        outliers_gt = (test_segmentation == 13).int()

        test_image = test_image.to(device).unsqueeze(0)
        test_logits = model(test_image)
        predicted_segmentation = (
            torch.argmax(test_logits, dim=1).squeeze(0).detach().cpu()
        )

        aupr = compute_aupr(outliers_gt.detach().cpu(), outliers_map.detach().cpu())
        miou = compute_mIoU(
            test_segmentation.detach().cpu(),
            predicted_segmentation.detach().cpu(),
            13 + 1,
        )

        metrics["aupr"].append(aupr)
        metrics["miou"].append(miou)

    results = {}
    for key in metrics.keys():
        results[key] = {}
        results[key]["mean"] = sum(metrics[key]) / len(metrics[key])
        results[key]["std"] = torch.std(torch.tensor(metrics[key])).item()

    return results
