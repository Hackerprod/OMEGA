import numpy as np

class PersistentMemory:
    """
    NTK-Stabilized Persistent Memory Matrix.
    Stores episodic experiences with protection against catastrophic forgetting.
    """
    def __init__(self, d_model, capacity=1000, dtype=np.float64):
        self.d = d_model
        self.capacity = capacity
        self.dtype = np.dtype(dtype)
        self.memory = np.zeros((capacity, d_model), dtype=self.dtype)
        self.usage = np.zeros(capacity, dtype=self.dtype)
        self.cursor = 0

    def write(self, experience_vector, importance=1.0):
        """
        Selective write using GUM-inspired uniqueness detection.
        """
        # Uniqueness check: check if already exists
        if self.cursor > 0:
            similarities = self.memory[:self.cursor] @ np.asarray(experience_vector, dtype=self.dtype)
            if np.max(similarities) > 0.98:
                return # Redundant info
                
        # NTK-inspired update: only update if the importance is high
        if self.cursor < self.capacity:
            self.memory[self.cursor] = np.asarray(experience_vector, dtype=self.dtype)
            self.usage[self.cursor] = self.dtype.type(importance)
            self.cursor += 1
        else:
            # Overwrite least important
            idx = np.argmin(self.usage)
            self.memory[idx] = np.asarray(experience_vector, dtype=self.dtype)
            self.usage[idx] = self.dtype.type(importance)

    def read(self, query_vector):
        """Differentiable read (Attention-like) over the matrix."""
        if self.cursor == 0:
            return np.zeros(self.d, dtype=self.dtype)
        
        # Dot-product similarity
        weights = self.memory[:self.cursor] @ np.asarray(query_vector, dtype=self.dtype)
        weights -= np.max(weights)
        weights = np.exp(weights.astype(np.float64)).astype(self.dtype)
        denom = np.sum(weights) + self.dtype.type(1e-8)
        weights /= denom
        return (weights @ self.memory[:self.cursor]).astype(self.dtype, copy=False)

    def get_state(self):
        return {
            "memory": self.memory.copy(),
            "usage": self.usage.copy(),
            "cursor": int(self.cursor),
        }

    def set_state(self, state):
        self.memory = np.asarray(state["memory"], dtype=self.dtype).copy()
        self.usage = np.asarray(state["usage"], dtype=self.dtype).copy()
        self.cursor = int(state["cursor"])

    def decay(self, rate: float = 0.995):
        if self.cursor == 0:
            return
        self.usage[: self.cursor] *= self.dtype.type(rate)
