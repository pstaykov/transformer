import numpy as np


class CrossEntropyLoss:
    """Softmax cross-entropy over the vocabulary dimension, for next-token prediction.

    Args:
        ignore_index: target value to exclude from the loss (e.g. a padding id).
    """

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.probs = None
        self.targets = None
        self.mask = None
        self.n_valid = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, T, V) float array
            targets: (B, T) int array of target token ids

        Returns:
            scalar mean cross-entropy loss over non-ignored positions
        """
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
        else:
            mask = np.ones_like(targets, dtype=bool)

        safe_targets = np.where(mask, targets, 0)
        target_probs = np.take_along_axis(probs, safe_targets[..., None], axis=-1)[..., 0]
        log_probs = -np.log(np.clip(target_probs, 1e-12, None))

        self.probs = probs
        self.targets = safe_targets
        self.mask = mask
        self.n_valid = max(int(np.sum(mask)), 1)

        return float(np.sum(log_probs * mask) / self.n_valid)

    def backward(self):
        """Returns dL/dlogits, shape (B, T, V)."""
        B, T, V = self.probs.shape
        dlogits = self.probs.copy()

        b_idx, t_idx = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
        dlogits[b_idx, t_idx, self.targets] -= 1

        dlogits *= self.mask[..., None]
        dlogits /= self.n_valid
        return dlogits
