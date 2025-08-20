import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans


def batch_nt_xent(x, target_matrix, temperature, device="cpu"):
    B = x.size(0)
    assert len(x.size()) == 3
    assert x.size(0) == len(target_matrix)

    loss = None
    for i in range(B):
        if loss is None:
            loss = nt_xent(x[i], target_matrix[i], temperature, device)
        else:
            loss += nt_xent(x[i], target_matrix[i], temperature, device)

    loss /= B
    return loss


# NT-Xent Loss
def nt_xent(x, target_matrix, temperature, device="cpu"):
    # pos_indices = pos_indices.to(device)
    x = x.to(device)
    assert len(x.size()) == 2

    # Cosine similarity
    xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1).to(device)
    # Set logit of diagonal element to "inf" signifying complete
    # correlation. sigmoid(inf) = 1.0 so this will work out nicely
    # when computing the Binary cross-entropy Loss.
    xcs[torch.eye(x.size(0)).bool().to(device)] = float("inf")

    # Standard binary cross-entropy loss. We use binary_cross_entropy() here and not
    # binary_cross_entropy_with_logits() because of
    # https://github.com/pytorch/pytorch/issues/102894
    # The method *_with_logits() uses the log-sum-exp-trick, which causes inf and -inf values
    # to result in a NaN result.
    # Convert target matrix to float
    target_matrix = target_matrix.float().to(device)
    loss = F.binary_cross_entropy(
        (xcs / temperature).sigmoid(), target_matrix, reduction="none"
    )

    target_pos = target_matrix.bool()
    target_neg = ~target_pos

    loss_pos = (
        torch.zeros(x.size(0), x.size(0))
        .to(device)
        .masked_scatter(target_pos, loss[target_pos])
        .to(device)
    )
    loss_neg = (
        torch.zeros(x.size(0), x.size(0))
        .to(device)
        .masked_scatter(target_neg, loss[target_neg])
        .to(device)
    )
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    num_pos = target_matrix.sum(dim=1)
    num_neg = x.size(0) - num_pos

    return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()


def nt_xent_loss(x, selected_pixels, target_matrix, temperature, device="cpu"):
    B, emb_size, H, W = x.shape
    x = x.view(B, emb_size, H * W)
    # B, emb_size, HW = x.shape

    x = torch.permute(x, (0, 2, 1))

    x_selected = torch.zeros(
        (B, selected_pixels.shape[1], emb_size), dtype=torch.float32
    )
    # selected_pixels = torch.unique(pos_pairs.flatten())
    for i in range(B):
        x_selected[i] = x[i, selected_pixels[i], :]
    x = x_selected
    return batch_nt_xent(x, target_matrix, temperature, device)


class NT_Xent_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NT_Xent_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, X, selected_pixels, target_matrix, **kwargs):
        return nt_xent_loss(X, selected_pixels, target_matrix, self.temperature)


def binarize(T, num_classes):
    device = T.device
    T = T.cpu().numpy()
    import sklearn.preprocessing

    T = sklearn.preprocessing.label_binarize(T, classes=range(0, num_classes))
    print(device)
    # if device == "cpu":
    if device == torch.device("cpu"):
        T = torch.FloatTensor(T)
    else:
        T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


from dataset import sample_pixels_per_class


class Proxy_Anchor(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_size,
        num_pixels_per_class=10,
        margin=0.1,
        alpha=32,
        class_weights=None,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if device == torch.device("cpu"):
            self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size))
        else:
            self.proxies = torch.nn.Parameter(
                torch.randn(num_classes, embedding_size).cuda()
            )

        # nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        # Manually normalize the proxies right after initialization
        self.proxies.data = torch.nn.functional.normalize(self.proxies.data, p=2, dim=1)

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.alpha = alpha
        self.num_pixels_per_class = num_pixels_per_class
        self.device = device

        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights, device=device, dtype=torch.float32
            )
        else:
            self.class_weights = torch.ones(num_classes, device=device)

    def forward(self, X, labels, **kwargs):
        # X: (B, C, H, W)
        # labels: (B, H, W)

        # Make:
        # X: (C, B*H*W)
        # labels: (B*H*W)
        print(X.shape)
        X = X.reshape(X.size(1), -1)
        labels = labels.reshape(-1)

        X, labels = sample_pixels_per_class(X, labels, self.num_pixels_per_class)

        # Pick num_pixels random pixels
        P = self.proxies

        cos = F.linear(l2_norm(X.T), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=labels, num_classes=self.num_classes).to(self.device)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.margin)).to(self.device)
        neg_exp = torch.exp(self.alpha * (cos + self.margin)).to(self.device)

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(
            P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp).to(self.device)
        ).sum(dim=0)
        N_sim_sum = torch.where(
            N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp).to(self.device)
        ).sum(dim=0)

        pos_term = (
            torch.log(1 + P_sim_sum) * self.class_weights
        ).sum() / num_valid_proxies
        neg_term = (
            torch.log(1 + N_sim_sum) * self.class_weights
        ).sum() / self.num_classes
        loss = pos_term + neg_term

        return loss


# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.clamp(d_ap - d_an + margin, min=0.0)
    return loss.mean()


import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class MagnetLoss(nn.Module):
    def __init__(
        self, num_classes, num_clusters=3, margin=0.1, cluster_update_every=10
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.margin = margin
        self.cluster_update_every = cluster_update_every

        self.cluster_centers = (
            None  # [M, D], M = total clusters across classes present in the batch
        )
        self.cluster_labels = None  # [N], global cluster index per sample
        self.epoch_counter = 0

    def update_clusters(self, X, labels, **kwargs):
        """
        embeddings: [N, D]
        labels:     [N]
        """
        device = X.device
        D = X.size(1)

        centers_list = []
        # we'll fill every position in this tensor (since every class with >0 samples is assigned)
        cluster_assignments = torch.empty(
            labels.size(0), dtype=torch.long, device=device
        )

        # global running index over all clusters created so far
        idx = 0

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            n_cls = int(cls_mask.sum().item())
            if n_cls == 0:
                # no samples of this class in the batch → skip (don't pad with zeros)
                continue

            # keep the feature dimensionality intact: [n_cls, D]
            cls_embeddings = X[cls_mask].detach().cpu().numpy()

            # if n_cls < num_clusters, cap k to n_cls
            k = min(self.num_clusters, n_cls)

            kmeans = KMeans(n_clusters=k, n_init=5)
            assignments = kmeans.fit_predict(cls_embeddings)  # [n_cls]
            centers = torch.tensor(
                kmeans.cluster_centers_,  # [k, D]
                device=device,
                dtype=X.dtype,
            )

            centers_list.append(centers)

            # assign global cluster ids to the samples of this class
            cluster_assignments[cls_mask] = (
                torch.as_tensor(assignments, device=device, dtype=torch.long) + idx
            )

            # advance global cluster index by the number of clusters we actually used
            idx += k

        if len(centers_list) == 0:
            # extremely degenerate case: empty batch
            self.cluster_centers = torch.zeros((1, D), device=device, dtype=X.dtype)
        else:
            self.cluster_centers = torch.cat(centers_list, dim=0)  # [M, D]
        self.cluster_labels = cluster_assignments  # [N]

    def forward(self, X, labels, **kwargs):
        """
        embeddings: [N, D]
        labels:     [N]
        """
        N, D = X.shape

        # Normalize embeddings to keep distances bounded
        X = torch.nn.functional.normalize(X, p=2, dim=1)

        if (self.cluster_centers is None) or (
            self.epoch_counter % self.cluster_update_every == 0
        ):
            self.update_clusters(X, labels)

        # Normalize cluster centers as well
        self.cluster_centers = torch.nn.functional.normalize(
            self.cluster_centers, p=2, dim=1
        )

        # Distances to all cluster centers
        dists = torch.cdist(X, self.cluster_centers) ** 2  # [N, M]

        # Clamp distances to avoid overflow/underflow
        dists = torch.clamp(dists, min=1e-6, max=50.0)

        # Distance to the assigned (positive) cluster for each sample
        pos_dists = dists[torch.arange(N, device=X.device), self.cluster_labels]  # [N]

        # Magnet loss denominator over all clusters
        # Use log-sum-exp trick for numerical stability
        logits = -0.5 * dists  # [N, M]
        pos_logits = -0.5 * pos_dists  # [N]

        # logsumexp across clusters
        log_denominator = torch.logsumexp(logits, dim=1)  # [N]

        # final loss
        loss = -(pos_logits - log_denominator)

        self.epoch_counter += 1
        return loss.mean()
