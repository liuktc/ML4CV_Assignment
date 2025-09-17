import torch
import torch.nn as nn


class NormalizedWeighting:
    def __init__(self, alpha=0.99):
        self.avg_metric = None
        self.avg_ce = None
        self.alpha = alpha

    def __call__(self, loss_metric, loss_ce):
        if self.avg_metric is None:
            self.avg_metric = loss_metric.item()
            self.avg_ce = loss_ce.item()
        else:
            self.avg_metric = (
                self.alpha * self.avg_metric + (1 - self.alpha) * loss_metric.item()
            )
            self.avg_ce = self.alpha * self.avg_ce + (1 - self.alpha) * loss_ce.item()

        norm_metric = loss_metric / (self.avg_metric + 1e-8)
        norm_ce = loss_ce / (self.avg_ce + 1e-8)
        return norm_metric + norm_ce


# (c) Uncertainty weighting (Kendall et al.)
class UncertaintyWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        # log_vars are log(sigma^2), start at 0
        self.log_var_metric = nn.Parameter(torch.zeros(1))
        self.log_var_ce = nn.Parameter(torch.zeros(1))

    def forward(self, loss_metric, loss_ce):
        precision_metric = torch.exp(-self.log_var_metric)
        precision_ce = torch.exp(-self.log_var_ce)

        loss = precision_metric * loss_metric + precision_ce * loss_ce
        loss += 0.5 * (self.log_var_metric + self.log_var_ce)
        return loss


class FixedWeighting(nn.Module):
    def __init__(self, lambda_metric=1.0, lambda_ce=1.0):
        super().__init__()
        self.lambda_metric = lambda_metric
        self.lambda_ce = lambda_ce

    def __call__(self, loss_metric, loss_ce):
        return self.lambda_metric * loss_metric + self.lambda_ce * loss_ce
