# utils.py

import torch
import numpy as np

class RunningStats:
    """Correct Welford online statistics algorithm implementation"""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences, not variance
        
    def update(self, x):
        """Update statistics, x can be a single value or batch data"""
        if np.isscalar(x):
            x = np.array([x])
        elif isinstance(x, (list, tuple)):
            x = np.array(x)
        
        # Flatten to 1D array
        x_flat = x.flatten()
        
        for value in x_flat:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.M2 += delta * delta2
            
    @property
    def variance(self):
        """Overall variance"""
        return self.M2 / self.count if self.count > 0 else 0.0
        
    @property
    def sample_variance(self):
        """Sample variance"""  
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0
        
    @property
    def std(self):
        """Standard deviation"""
        return np.sqrt(self.variance)
        
    def normalize(self, x):
        """Normalize data"""
        return (x - self.mean) / (self.std + 1e-8)