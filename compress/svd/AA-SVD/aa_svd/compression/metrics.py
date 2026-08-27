from typing import Tuple

import torch
import torch.nn.functional as F


class RunningStat:
    """Numerically stable running mean and variance using Welford's parallel batch update."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # sum of squared deviations from the mean

    def update_batch(self, values: torch.Tensor) -> None:
        """Merge a batch of scalar values into the running statistics.

        Equivalent to processing each element individually but computed in one pass.
        *values* should be a 1-D tensor of shape (B,).
        """
        values = values.detach()
        batch_count = values.numel()
        batch_mean = values.mean().item()
        batch_M2 = values.var(unbiased=False).item() * batch_count

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.M2 = batch_M2
            return

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / new_count)
        self.M2 = self.M2 + batch_M2 + delta ** 2 * self.count * batch_count / new_count
        self.count = new_count

    def compute(self) -> Tuple[float, float]:
        """Return (mean, std) over all values seen so far."""
        if self.count == 0:
            return float("nan"), float("nan")
        return self.mean, (self.M2 / self.count) ** 0.5


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape *x* from (..., hidden_dim) to (N, hidden_dim)."""
    if x.dim() == 2:
        return x
    return x.reshape(-1, x.shape[-1])


class MSEMetric:
    """Mean squared error per sample, aggregated with running statistics."""

    def __init__(self) -> None:
        self.stats = RunningStat()

    def update(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        y, y_hat = ensure_2d(y), ensure_2d(y_hat)
        self.stats.update_batch((y - y_hat).pow(2).mean(dim=1))

    def compute(self) -> Tuple[float, float]:
        return self.stats.compute()


class CosineDistanceMetric:
    """Cosine distance (1 - cosine similarity) per sample, aggregated with running statistics."""

    def __init__(self) -> None:
        self.stats = RunningStat()

    def update(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        y, y_hat = ensure_2d(y), ensure_2d(y_hat)
        self.stats.update_batch(1 - F.cosine_similarity(y, y_hat, dim=1))

    def compute(self) -> Tuple[float, float]:
        return self.stats.compute()


class NormComparatorMetric:
    """Fraction of samples where the compressed output norm exceeds the original norm."""

    def __init__(self) -> None:
        self.stats = RunningStat()

    def update(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        y, y_hat = ensure_2d(y), ensure_2d(y_hat)
        higher = (torch.norm(y_hat, dim=1) > torch.norm(y, dim=1)).float()
        self.stats.update_batch(higher)

    def compute(self) -> Tuple[float, float]:
        return self.stats.compute()


class PerplexityMetric:
    """Perplexity estimated from cross-entropy loss, aggregated with running statistics."""

    def __init__(self) -> None:
        self.stats = RunningStat()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate cross-entropy loss values.

        Args:
            logits:  (B, vocab_size) or (B, S, vocab_size) — will be flattened.
            targets: (B,) integer token ids, or (B, vocab_size) soft probability distribution.
        """
        logits = logits.view(-1, logits.size(-1))
        if targets.ndim == 2:
            targets = targets.view(-1)
        else:
            targets = targets.softmax(dim=-1).view(-1, targets.size(-1))
        self.stats.update_batch(F.cross_entropy(logits, targets, reduction='none'))

    def compute(self, exp: bool = True) -> float:
        """Return mean perplexity (exp=True) or mean cross-entropy loss (exp=False)."""
        mean, _ = self.stats.compute()
        if exp:
            return torch.exp(torch.tensor(mean)).item()
        return mean
