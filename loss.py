import torch
import torch.nn.functional as F
import torch.nn as nn


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
    xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
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
    target_matrix = target_matrix.float()
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


def binarize(T, nb_classes):
    device = T.device
    T = T.cpu().numpy()
    import sklearn.preprocessing

    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
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


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, device="cpu"):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if device == "cpu":
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        else:
            self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())

        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss
