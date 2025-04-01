import numpy as np

class Normalize():
    def __init__(
        self,
        v_min,
        v_max,
    ):
        self.v_min = np.array(v_min, dtype=np.float32)
        self.v_max = np.array(v_max, dtype=np.float32)
        
    def normalize_forw(self, v):
        normalized = (2.0 * v - self.v_min - self.v_max) / (self.v_max - self.v_min)
        return normalized

    def normalize_back(self, v):
        denomalized = 0.5 * (self.v_min + self.v_max + (self.v_max - self.v_min) * v)
        return denomalized
    
class Normalize_gaussian():
    def __init__(
        self,
        mean,
        std,
    ):
        self.mean = mean
        self.std = std
        
    def normalize_forw(self, v):
        normalized = (v - self.mean) / self.std
        return normalized

    def normalize_back(self, v):
        denomalized = v * self.std + self.mean
        return denomalized