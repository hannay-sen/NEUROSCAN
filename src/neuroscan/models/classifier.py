"""EfficientNet-B0 classifier with temperature scaling, MC-dropout and Mahalanobis OOD."""
from __future__ import annotations

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TumorClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "efficientnet_b0",
                 pretrained: bool = False, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=num_classes, drop_rate=drop_rate)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        # Mahalanobis statistics (filled by fit_ood); empty means "not fitted".
        self.register_buffer("ood_means", torch.empty(0))
        self.register_buffer("ood_precision", torch.empty(0))
        self.register_buffer("ood_threshold", torch.tensor(float("nan")))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # OOD buffers are empty until fitted; resize them so checkpoints load cleanly.
        for k in ("ood_means", "ood_precision", "ood_threshold"):
            if k in state_dict:
                setattr(self, k, torch.empty_like(state_dict[k]))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ---- forward passes -------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)

    # ---- temperature scaling (Guo et al., 2017) -------------------------
    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)
        logits, labels = logits.detach().cpu(), labels.cpu()

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / log_t.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        t = float(log_t.detach().exp().clamp(0.05, 20.0))
        self.temperature.data.fill_(t)
        return t

    # ---- Mahalanobis OOD (Lee et al., 2018), tied covariance -----------
    def fit_ood(self, train_feats: torch.Tensor, train_labels: torch.Tensor,
                val_feats: torch.Tensor, percentile: float = 99.0) -> float:
        f, y = train_feats.double().cpu(), train_labels.cpu()
        classes = torch.unique(y)
        means = torch.stack([f[y == c].mean(0) for c in classes])
        centered = f - means[torch.searchsorted(classes, y)]
        cov = centered.T @ centered / len(f)
        cov += torch.eye(cov.shape[0], dtype=cov.dtype) * 1e-3 * cov.diagonal().mean()
        self.ood_means = means.float()
        self.ood_precision = torch.linalg.inv(cov).float()
        scores = self.mahalanobis(val_feats)
        thr = float(np.percentile(scores.numpy(), percentile))
        self.ood_threshold = torch.tensor(thr)
        return thr

    def ood_fitted(self) -> bool:
        return self.ood_means.numel() > 0 and not math.isnan(float(self.ood_threshold))

    @torch.no_grad()
    def mahalanobis(self, feats: torch.Tensor) -> torch.Tensor:
        f = feats.float().cpu()
        diffs = f[:, None, :] - self.ood_means.cpu()[None]            # N x C x D
        d = torch.einsum("ncd,de,nce->nc", diffs, self.ood_precision.cpu(), diffs)
        return d.min(dim=1).values.clamp_min(0).sqrt()

    # ---- inference -----------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, mc_samples: int = 20) -> dict:
        """x: 1x3xHxW. Returns calibrated probs, MC-dropout entropy, OOD score."""
        self.eval()
        feats = self.features(x)
        logits = self.backbone.get_classifier()(feats)
        probs = self.calibrated_probs(logits)[0]

        # MC-dropout (Gal & Ghahramani, 2016): EfficientNet's only dropout sits between the
        # pooled features and the classifier head, so we resample just that step.
        head = self.backbone.get_classifier()
        p = self._drop_rate()
        mc = torch.stack([self.calibrated_probs(head(F.dropout(feats, p=p, training=True)))[0]
                          for _ in range(max(mc_samples, 1))])
        mean_p = mc.mean(0)
        k = mean_p.numel()
        entropy = float(-(mean_p * (mean_p + 1e-12).log()).sum() / math.log(k))  # normalized 0..1

        ood_score = float(self.mahalanobis(feats)[0]) if self.ood_fitted() else None
        return {
            "probs": probs.cpu().numpy(),
            "mc_mean_probs": mean_p.cpu().numpy(),
            "entropy": entropy,
            "ood_score": ood_score,
            "ood_threshold": float(self.ood_threshold) if self.ood_fitted() else None,
        }

    def _drop_rate(self) -> float:
        return float(getattr(self.backbone, "drop_rate", 0.3) or 0.3)
