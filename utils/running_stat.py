import numpy as np

class RunningMeanStd:
    """
    Runtime online mean/variance (Welford) + normalize to z-score, then compress to [-10, 10] with tanh as learning signal.
    - Interface accepts numpy arrays, returns numpy arrays of the same shape
    - Initial variance set to 1 to avoid excessive scaling early on
    """
    def __init__(self, eps: float = 1e-8):
        self.count = 1e-4  # Avoid division by zero
        self.mean = 0.0
        self.M2 = 1.0  # Initialize std ≈ 1
        self.eps = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n == 0:
            return
        # Batch Welford
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = float(n)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        # Combine variances
        m2 = self.M2 + batch_var * batch_count + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.M2 = m2
        self.count = total_count

    @property
    def var(self) -> float:
        return self.M2 / max(self.count, 1.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, self.eps)))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / (self.std + self.eps)
        # Clip to [-10, 10]
        z = 10.0 * np.tanh(z)
        return z.astype(np.float32)
