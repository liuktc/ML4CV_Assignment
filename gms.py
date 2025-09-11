import torch
import torch.nn as nn
import torch.distributions as dist


class ClassConditionalGMM(nn.Module):
    """
    A class-conditional Gaussian Mixture Model detector.

    For each class c in {0,…,C-1}, there is a GMM with K components (either full or diag covariance).
    Provides:
      - fit(X, y): standard EM fit on the whole data
      - fit_batch(X, y): accumulate statistics from a batch, can be called multiple times, then finalize by calling finalize_batch()
      - predict_proba(X): P(y = c | x)
      - predict(X): argmax class
      - optional outlier score etc.

    Note: after using fit_batch on all data batches, you must call finalize_batch() (or internally in predict if needed).
    """

    def __init__(
        self,
        num_classes,
        n_components,
        dim,
        covariance_type="full",
        reg_covar=1e-6,
        device=None,
        seed=12345,
    ):
        """
        Args:
          num_classes: number of classes C
          n_components: number of mixture components per class K
          dim: dimension of data
          covariance_type: 'full' or 'diag'
          reg_covar: small regularization added to diagonal of covariances
        """
        super().__init__()
        # Seed everything for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.num_classes = num_classes
        self.n_components = n_components
        self.dim = dim
        assert covariance_type in ("full", "diag"), (
            "covariance_type must be 'full' or 'diag'"
        )
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar

        # parameters per class c, component k:
        # mixture weights, means, covariances
        # weights: shape (C, K)
        self.weights = nn.Parameter(
            torch.ones(num_classes, n_components) / n_components, requires_grad=False
        )
        # means: (C, K, dim)
        self.means = nn.Parameter(
            torch.randn(num_classes, n_components, dim), requires_grad=False
        )
        # covariances:
        if covariance_type == "full":
            # covariances: (C, K, dim, dim)
            eye = torch.eye(dim)
            self.covariances = nn.Parameter(
                torch.stack(
                    [
                        torch.stack([eye for _ in range(n_components)])
                        for _ in range(num_classes)
                    ]
                ),
                requires_grad=False,
            )
        else:  # diag
            # covariances: (C, K, dim) -> only diagonal entries
            self.covariances = nn.Parameter(
                torch.ones(num_classes, n_components, dim), requires_grad=False
            )

        # class prior counts
        self.class_counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
        self.total_count = 0.0

        # For batch fitting: sufficient statistics placeholders
        self._ss_initialized = False
        # Sufficient stats per class, per component:
        # Nk[c, k], sum[c, k, :] (over x), sum_outer[c, k, :, :] (for full)
        # sum_vardiag[c, k, :] (for diag)
        self.ss_Nk = None
        self.ss_sum = None
        self.ss_sum_outer = None  # full
        self.ss_sum_vardiag = None  # diag

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def _init_sufficient_stats(self):
        C, K, d = self.num_classes, self.n_components, self.dim
        self.ss_Nk = torch.zeros(C, K, dtype=torch.float64, device=self.device)
        self.ss_sum = torch.zeros(C, K, d, dtype=torch.float64, device=self.device)
        if self.covariance_type == "full":
            self.ss_sum_outer = torch.zeros(
                C, K, d, d, dtype=torch.float64, device=self.device
            )
            self.ss_sum_vardiag = None
        else:
            self.ss_sum_vardiag = torch.zeros(
                C, K, d, dtype=torch.float64, device=self.device
            )
            self.ss_sum_outer = None
        self.ss_class_counts = torch.zeros(C, dtype=torch.float64, device=self.device)
        self.ss_total_count = 0.0
        self._ss_initialized = True

    def fit(self, X, y, n_iter=10, verbose=False):
        """
        Fit GMMs for each class on full data (all of X, y).
        X: tensor (N, d)
        y: tensor (N,) with integers in 0..num_classes-1
        """
        X = X.to(self.device)
        y = y.to(self.device)
        N, d = X.shape
        C, K = self.num_classes, self.n_components

        # initialize parameters: you could do KMeans per class for means, here we just random or use existing
        # For simplicity: re-init means randomly from data within each class
        with torch.no_grad():
            for c in range(C):
                mask = y == c
                Nc = mask.sum().item()
                if Nc == 0:
                    continue
                Xc = X[mask]
                # pick K samples (with replacement if Nc<K)
                inds = torch.randperm(Nc, device=self.device)
                if Nc >= K:
                    chosen = Xc[inds[:K]]
                else:
                    # repeat
                    reps = K // Nc + 1
                    Xc_rep = Xc.repeat(reps, 1)
                    chosen = Xc_rep[:K]
                self.means[c].data.copy_(chosen)
                # weights uniform
                self.weights[c].data.copy_(torch.ones(K, device=self.device) / K)
                # covariances init: identity (or diag 1)
                if self.covariance_type == "full":
                    eye = torch.eye(d, device=self.device)
                    for k in range(K):
                        self.covariances[c, k].data.copy_(eye)
                else:
                    self.covariances[c].data.copy_(torch.ones(K, d, device=self.device))

        self.class_counts = torch.zeros(C, dtype=torch.float64, device=self.device)
        self.total_count = 0.0

        for it in range(n_iter):
            # E-step: compute responsibilities for all classes and components
            # For each class c, compute log p(x | class=c, component k) + log weight
            # We'll also need class prior, but for fitting each class separately we just look at class-specific data
            # So simpler: for each class c, restrict to Xc
            resp = {}  # resp[(c)] = tensor (Nc, K)
            Nk = torch.zeros(C, K, dtype=torch.float64, device=self.device)
            sum_c = torch.zeros(C, K, d, dtype=torch.float64, device=self.device)
            if self.covariance_type == "full":
                sum_outer = torch.zeros(
                    C, K, d, d, dtype=torch.float64, device=self.device
                )
            else:
                sum_vardiag = torch.zeros(
                    C, K, d, dtype=torch.float64, device=self.device
                )

            class_counts = torch.zeros(C, dtype=torch.float64, device=self.device)

            for c in range(C):
                mask = y == c
                Xc = X[mask]
                Nc = Xc.shape[0]
                class_counts[c] = Nc
                if Nc == 0:
                    continue

                # compute log probs under each component
                # shape (Nc, K)
                log_probs_ck = self._compute_log_prob(
                    Xc, class_idx=c
                )  # includes log weight
                # normalize to get responsibilities
                # logsumexp over k
                max_lp, _ = torch.max(log_probs_ck, dim=1, keepdim=True)
                lse = max_lp + torch.log(
                    torch.exp(log_probs_ck - max_lp).sum(dim=1, keepdim=True)
                )
                log_resp_ck = log_probs_ck - lse  # (Nc, K)
                resp_ck = torch.exp(log_resp_ck)

                Nk[c] = resp_ck.sum(dim=0).double()  # shape (K,)
                sum_c[c] = resp_ck.t().double() @ Xc.double()  # (K, d)

                if self.covariance_type == "full":
                    for k in range(K):
                        X_center = Xc.double() - self.means[c, k].double()
                        # (Nc, d); outer product
                        weighted = resp_ck[:, k].unsqueeze(1) * X_center
                        # sum of outer products
                        so = weighted.t() @ X_center  # (d, d)
                        sum_outer[c, k] = so
                else:
                    # diag variant
                    # compute per-component diag variance
                    # E[ (x − μ)^2 ] weight by resp
                    diff2 = (
                        Xc.double().unsqueeze(1)
                        - self.means[c, :].double().unsqueeze(0)
                    ) ** 2  # (Nc, K, d)
                    # resp_ck.unsqueeze(2) * diff2 sum over Nc
                    sum_vardiag[c] = (resp_ck.unsqueeze(2).double() * diff2).sum(dim=0)

            # M-step: update parameters per class, component

            with torch.no_grad():
                for c in range(C):
                    Nc = class_counts[c]
                    self.class_counts[c] = Nc
                self.total_count = class_counts.sum().item()

                for c in range(C):
                    if class_counts[c] == 0:
                        continue
                    for k in range(K):
                        Nk_ck = Nk[c, k]
                        if Nk_ck < 1e-8:
                            # avoid zeros; leave parameters as is or small perturbation
                            continue
                        # update weight
                        self.weights[c, k] = Nk_ck / class_counts[c]
                        # update mean
                        self.means[c, k] = (sum_c[c, k] / Nk_ck).to(self.device)

                    # update covariances
                    if self.covariance_type == "full":
                        for k in range(K):
                            Nk_ck = Nk[c, k]
                            if Nk_ck < 1e-8:
                                continue
                            # E[ xx^T ] − μμ^T
                            cov = sum_outer[c, k] / Nk_ck - torch.ger(
                                self.means[c, k].double(), self.means[c, k].double()
                            )
                            cov = cov.double()
                            # ensure symmetry
                            cov = (cov + cov.t()) / 2.0
                            # eigen decomposition
                            eigs, vecs = torch.linalg.eigh(cov)
                            eigs_clipped = torch.clamp(eigs, min=self.reg_covar)
                            cov_pd = vecs @ torch.diag(eigs_clipped) @ vecs.T
                            cov = cov_pd
                            # regularize
                            cov = cov + self.reg_covar * torch.eye(
                                self.dim, dtype=torch.float64, device=self.device
                            )
                            self.covariances[c, k] = cov.to(self.device)
                    else:
                        for k in range(K):
                            Nk_ck = Nk[c, k]
                            if Nk_ck < 1e-8:
                                continue
                            var_diag = sum_vardiag[c, k] / Nk_ck
                            # already E[(x−μ)^2]
                            var_diag = var_diag + self.reg_covar
                            self.covariances[c, k] = var_diag.to(self.device)

            if verbose:
                print(f"[fit] Iter {it + 1}/{n_iter} done.")
                ll = self.score_samples(X, y).item()
                print(f"[fit] Iter {it + 1}/{n_iter}, log-likelihood = {ll:.6f}")

    def fit_batch(self, X, y):
        """
        Accumulate statistics from one batch. Can be called many times over different subsets of data.
        Accumulates sufficient statistics; one must call finalize_batch(n_iter=..., verbose=...) to do EM using them.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        N, d = X.shape
        C, K = self.num_classes, self.n_components

        if not self._ss_initialized:
            self._init_sufficient_stats()

        # For each class c in this batch:
        for c in range(C):
            mask = y == c
            Xc = X[mask]
            Nc = Xc.shape[0]
            if Nc == 0:
                continue
            self.ss_class_counts[c] += Nc
            self.ss_total_count += Nc

            # compute responsibilities under current params
            log_probs_ck = self._compute_log_prob(
                Xc, class_idx=c
            )  # includes log weight
            max_lp, _ = torch.max(log_probs_ck, dim=1, keepdim=True)
            lse = max_lp + torch.log(
                torch.exp(log_probs_ck - max_lp).sum(dim=1, keepdim=True)
            )
            log_resp_ck = log_probs_ck - lse  # (Nc, K)
            resp_ck = torch.exp(log_resp_ck).double()

            self.ss_Nk[c] += resp_ck.sum(dim=0)  # (K,)
            self.ss_sum[c] += resp_ck.t() @ Xc.double()  # (K, d)

            if self.covariance_type == "full":
                for k in range(K):
                    X_center = Xc.double() - self.means[c, k].double()
                    wc = resp_ck[:, k].unsqueeze(1)
                    so = (wc * X_center).t() @ X_center  # (d, d)
                    self.ss_sum_outer[c, k] += so
            else:
                diff2 = (
                    Xc.double().unsqueeze(1) - self.means[c].double().unsqueeze(0)
                ) ** 2  # (Nc, K, d)
                self.ss_sum_vardiag[c] += (resp_ck.unsqueeze(2) * diff2).sum(dim=0)

        # Note: means, weights, covariances not updated here. Only accumulate stats.

    def finalize_batch(self, verbose=False):
        """
        After accumulating over all batches, perform M-step to update parameters.
        Optionally do multiple iterations of EM by repeated accumulation and M-step (but here just once).
        """
        if not self._ss_initialized:
            raise RuntimeError("Must call fit_batch at least once before finalizing")

        C, K, d = self.num_classes, self.n_components, self.dim

        # Update class counts etc
        with torch.no_grad():
            for c in range(C):
                self.class_counts[c] = self.ss_class_counts[c]
            self.total_count = self.ss_total_count

            # Update weights & means
            tot_means_change = torch.zeros(1).to(self.device)
            for c in range(C):
                Nc = self.ss_class_counts[c]
                if Nc == 0:
                    continue
                for k in range(K):
                    Nk_ck = self.ss_Nk[c, k]
                    if Nk_ck < 1e-8:
                        # leave old params
                        continue
                    # weight of component within class
                    self.weights[c, k] = Nk_ck / (Nc)
                    # mean
                    tot_means_change += torch.norm(
                        self.means[c, k].double().to(self.device)
                        - (self.ss_sum[c, k] / Nk_ck).to(self.device)
                    )
                    self.means[c, k] = (self.ss_sum[c, k] / Nk_ck).to(self.device)

            # Update covariances
            if self.covariance_type == "full":
                for c in range(C):
                    if self.ss_class_counts[c] == 0:
                        continue
                    for k in range(K):
                        Nk_ck = self.ss_Nk[c, k]
                        if Nk_ck < 1e-8:
                            continue
                        outer = self.ss_sum_outer[c, k] / Nk_ck
                        mean = self.means[c, k].double()
                        cov = outer - torch.ger(mean, mean)
                        cov = (cov + cov.t()) / 2.0
                        # eigen decomposition
                        eigs, vecs = torch.linalg.eigh(cov)
                        eigs_clipped = torch.clamp(eigs, min=self.reg_covar)
                        cov_pd = vecs @ torch.diag(eigs_clipped) @ vecs.T
                        cov = cov_pd
                        cov = cov + self.reg_covar * torch.eye(
                            d, dtype=torch.float64, device=self.device
                        )
                        self.covariances[c, k] = cov.to(self.device)
            else:
                tot = torch.zeros(1).to(self.device)
                for c in range(C):
                    if self.ss_class_counts[c] == 0:
                        continue
                    for k in range(K):
                        Nk_ck = self.ss_Nk[c, k]
                        if Nk_ck < 1e-8:
                            continue
                        var_diag = (self.ss_sum_vardiag[c, k] / Nk_ck) + self.reg_covar
                        # Change
                        tot += torch.norm(
                            var_diag.to(self.device)
                            - self.covariances[c, k].double().to(self.device)
                        )
                        self.covariances[c, k] = var_diag.to(self.device)
                print(
                    f"[finalize_batch] Average change in diag covariances: {tot.item() / (C * K):.6f}, {tot_means_change.item() / (C * K):.6f}"
                )

        if verbose:
            # some checks / tests: ensure consistency etc
            self._test_batch_vs_full()
        # Clear sufficient stats so new batches can be accumulated fresh
        self._ss_initialized = False

    def predict_proba(self, X, return_prob_dist=False):
        """
        Return P(y = c | x), shape (B, C)
        """
        X = X.to(self.device)
        B, d = X.shape
        C, K = self.num_classes, self.n_components

        log_ps = []
        # class prior: P(y=c)
        class_prior = (self.class_counts + 1e-8) / (self.total_count + 1e-8 * C)
        log_prior = torch.log(class_prior.to(self.device))

        for c in range(C):
            # log p(x | class c)
            log_px_c = self._score_samples_class(X, class_idx=c)  # (B,)
            log_ps.append(log_px_c + log_prior[c])

        log_ps = torch.stack(log_ps, dim=1)  # (B, C)
        # normalize
        max_lp, _ = torch.max(log_ps, dim=1, keepdim=True)
        ps = torch.exp(log_ps - max_lp)
        if return_prob_dist:
            ps = ps / ps.sum(dim=1, keepdim=True)

        return ps

    def predict(self, X):
        ps = self.predict_proba(X)
        return torch.argmax(ps, dim=1)

    def score_samples(self, X, y=None):
        """
        Returns log likelihood of the data under the model.
        If y is given, returns sum over classes (weighted by prior) or only class-conditional.
        Here returning total log-likelihood over all samples: log p(x, y)
        """
        # log p(x, y) = log p(y) + log p(x | y)
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)
            lp = torch.zeros_like(y, dtype=torch.float64, device=self.device)
            ps = self.predict_proba(X)  # (N, C)
            # better: for each sample sum log prior + log likelihood of its true class
            class_prior = (self.class_counts + 1e-8) / (
                self.total_count + 1e-8 * self.num_classes
            )
            log_prior = torch.log(class_prior.to(self.device))
            for c in range(self.num_classes):
                mask = y == c
                if mask.sum() == 0:
                    continue
                lp[mask] = (
                    log_prior[c] + self._score_samples_class(X[mask], class_idx=c)
                ).to(lp.dtype)
            return lp.sum()
        else:
            # sum over all classes weighted by prior: log p(x) = logsumexp over classes
            ps = []
            class_prior = (self.class_counts + 1e-8) / (
                self.total_count + 1e-8 * self.num_classes
            )
            log_prior = torch.log(class_prior.to(self.device))
            for c in range(self.num_classes):
                lp_c = self._score_samples_class(X, class_idx=c) + log_prior[c]
                ps.append(lp_c.unsqueeze(1))
            log_ps = torch.cat(ps, dim=1)  # (N, C)
            # logsumexp over classes
            max_lp, _ = torch.max(log_ps, dim=1, keepdim=True)
            lse = max_lp + torch.log(
                torch.exp(log_ps - max_lp).sum(dim=1, keepdim=True)
            )
            return lse.sum()

    # internal helper methods

    def _compute_log_prob(self, X, class_idx):
        """
        Compute log p_k(x) for components k in class class_idx, plus log mixture weight
        Returns shape (B, K)
        """
        # X: (B, d)
        B, d = X.shape
        K = self.n_components
        means_ck = self.means[class_idx]  # (K, d)
        if self.covariance_type == "full":
            covs_ck = self.covariances[class_idx]  # (K, d, d)
        else:
            covs_ck = self.covariances[class_idx]  # (K, d) diag

        log_probs = []
        for k in range(K):
            mu = means_ck[k]
            if self.covariance_type == "full":
                cov = covs_ck[k] + self.reg_covar * torch.eye(d, device=self.device)
                mvn = dist.MultivariateNormal(mu, covariance_matrix=cov)
            else:
                # diag covariance
                cov_diag = covs_ck[k] + self.reg_covar
                mvn = dist.MultivariateNormal(
                    mu, covariance_matrix=torch.diag(cov_diag)
                )
            log_p = mvn.log_prob(X)  # (B,)
            # add log weight
            log_w = torch.log(self.weights[class_idx, k] + 1e-12)
            log_probs.append(log_p + log_w)
        log_probs = torch.stack(log_probs, dim=1)  # (B, K)
        return log_probs

    def _score_samples_class(self, X, class_idx):
        """
        Compute log p(x | class=class_idx) i.e. logsumexp over components
        Returns (B,)
        """
        log_probs = self._compute_log_prob(X, class_idx=class_idx)  # (B, K)
        max_lp, _ = torch.max(log_probs, dim=1, keepdim=True)
        lse = max_lp + torch.log(torch.exp(log_probs - max_lp).sum(dim=1, keepdim=True))
        return lse.squeeze(1)

    def _test_batch_vs_full(self, tol=1e-5):
        """
        Internal test: compare output of fitting via fit (on all data) versus
        fitting via many fit_batch + finalize_batch, on the same data.
        This test assumes that user has stored the full data examples somewhere
        in self._last_test_X, self._last_test_y
        If those are not set, no test is done.
        """
        if hasattr(self, "_last_test_X") and hasattr(self, "_last_test_y"):
            # Make a clone
            X = self._last_test_X.to(self.device)
            y = self._last_test_y.to(self.device)
            # Save current params
            weights_save = self.weights.clone()
            means_save = self.means.clone()
            covs_save = self.covariances.clone()

            # Do full fit into a fresh model
            m2 = ClassConditionalGMM(
                self.num_classes,
                self.n_components,
                self.dim,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                device=self.device,
            )
            m2.fit(
                X, y, n_iter=1, verbose=False
            )  # same as one M-step from full data (approx)
            # Do batch style
            # reset this model to initial params before batch
            with torch.no_grad():
                self.weights.copy_(weights_save)
                self.means.copy_(means_save)
                self.covariances.copy_(covs_save)
            self._init_sufficient_stats()
            # split into 2 batches
            N = X.shape[0]
            idx = torch.randperm(N, device=self.device)
            mid = N // 2
            X1, y1 = X[idx[:mid]], y[idx[:mid]]
            X2, y2 = X[idx[mid:]], y[idx[mid:]]
            self.fit_batch(X1, y1)
            self.fit_batch(X2, y2)
            self.finalize_batch(verbose=False)

            # Now compare parameters: weights, means, covariances
            w2, mu2, cov2 = m2.weights, m2.means, m2.covariances
            w_b, mu_b, cov_b = self.weights, self.means, self.covariances

            # Allow for permutation of components within a class: more complicated.
            # Here we assume initialization was same / random, so we compare loosely.

            max_diff = torch.max(torch.abs(w2 - w_b))
            if max_diff > tol:
                print(
                    f"[Test FAILED] weights differ: max absolute diff = {max_diff.item()}"
                )
            else:
                print(f"[Test PASSED] weights match within tol {tol}")

            max_diff = torch.max(torch.abs(mu2 - mu_b))
            if max_diff > tol:
                print(
                    f"[Test FAILED] means differ: max absolute diff = {max_diff.item()}"
                )
            else:
                print(f"[Test PASSED] means match within tol {tol}")

            # Covariance comparison: for full and diag separately
            max_diff = torch.max(torch.abs(cov2 - cov_b))
            if max_diff > tol * 10:  # covariance might differ a bit more
                print(
                    f"[Test FAILED] covariances differ: max absolute diff = {max_diff.item()}"
                )
            else:
                print(f"[Test PASSED] covariances match within tol×10 {tol * 10}")
        else:
            print(
                "No stored data for batch vs full test (_last_test_X, _last_test_y missing)"
            )

    def store_for_test(self, X, y):
        """
        Store data to be used for internal test of batch vs full
        """
        self._last_test_X = X.clone().detach()
        self._last_test_y = y.clone().detach()


if __name__ == "__main__":
    import random
    import numpy as np

    # make everything deterministic / reproducible
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # generate synthetic data
    N = 1000
    d = 2
    C = 2
    K = 3

    X0 = torch.randn(N, d) * 5 + torch.tensor([-3.0, 0.0])
    X1 = torch.randn(N, d) * 5 + torch.tensor([+3.0, 0.0])
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat(
        [torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)], dim=0
    )

    # Ensure the initial model parameters are the same for both
    # Create a “template” model with fixed initialization
    template = ClassConditionalGMM(
        num_classes=C, n_components=K, dim=d, covariance_type="full"
    )
    # force initial means/weights/covariances to reproducible values
    # you might manually initialize via torch.rand or copying, but due to the seed above
    # template.means, template.weights etc are already reproducible

    # Clone into two models: model_full (will use fit), model_batch (will use repeated fit_batch + finalize_batch)
    model_full = ClassConditionalGMM(
        num_classes=C, n_components=K, dim=d, covariance_type="full"
    )
    model_batch = ClassConditionalGMM(
        num_classes=C, n_components=K, dim=d, covariance_type="full"
    )

    # Copy parameters from template into both
    with torch.no_grad():
        model_full.means.copy_(template.means)
        model_full.weights.copy_(template.weights)
        model_full.covariances.copy_(template.covariances)

        model_batch.means.copy_(template.means)
        model_batch.weights.copy_(template.weights)
        model_batch.covariances.copy_(template.covariances)

    # Store data in both for the internal test to compare later
    model_full.store_for_test(X, y)
    model_batch.store_for_test(X, y)

    # Set same number of EM iterations
    n_em_iter = 50

    print("=== Full fit ===")
    model_full.fit(X, y, n_iter=n_em_iter, verbose=True)

    print("=== Batch fit ===")
    for it in range(n_em_iter):
        # split into batches (say two batches)
        # Use the same split every iteration for consistency
        mid = X.shape[0] // 2
        X1, y1 = X[:mid], y[:mid]
        X2, y2 = X[mid:], y[mid:]
        model_batch.fit_batch(X1, y1)
        model_batch.fit_batch(X2, y2)
        model_batch.finalize_batch(verbose=False)

    # Now compare parameters
    w_full, mu_full, cov_full = (
        model_full.weights,
        model_full.means,
        model_full.covariances,
    )
    w_batch, mu_batch, cov_batch = (
        model_batch.weights,
        model_batch.means,
        model_batch.covariances,
    )

    def max_abs(t):
        return float(torch.max(torch.abs(t)).item())

    print("Max abs diff in weights:", max_abs(w_full - w_batch))
    print("Max abs diff in means:", max_abs(mu_full - mu_batch))
    print("Max abs diff in covariances:", max_abs(cov_full - cov_batch))

    # Also test predictions are similar
    X_test = torch.tensor(
        [[-3.0, 0.0], [3.0, 0.0], [0.0, 0.0], [-1.0, 1.0], [1.0, -1.0]]
    )
    probs_full = model_full.predict_proba(X_test)
    probs_batch = model_batch.predict_proba(X_test)

    print("Predicted class probabilities (full fit):", probs_full)
    print("Predicted class probabilities (batch fit):", probs_batch)
    print("Max abs diff in predicted probabilities:", max_abs(probs_full - probs_batch))
    print("Predicted classes (full fit):", model_full.predict(X_test))
    print("Predicted classes (batch fit):", model_batch.predict(X_test))
