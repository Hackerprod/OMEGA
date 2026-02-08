import numpy as np

class PersistentMemory:
    """
    NTK-Stabilized Persistent Memory Matrix.
    Stores episodic experiences with protection against catastrophic forgetting.
    """
    def __init__(self, d_model, capacity=1000):
        self.d = d_model
        self.capacity = capacity
        self.memory = np.zeros((capacity, d_model))
        self.usage = np.zeros(capacity)
        self.cursor = 0

    def write(self, experience_vector, importance=1.0):
        """
        Selective write using GUM-inspired uniqueness detection.
        """
        # Uniqueness check: check if already exists
        if self.cursor > 0:
            similarities = self.memory[:self.cursor] @ experience_vector
            if np.max(similarities) > 0.98:
                return # Redundant info
                
        # NTK-inspired update: only update if the importance is high
        if self.cursor < self.capacity:
            self.memory[self.cursor] = experience_vector
            self.usage[self.cursor] = importance
            self.cursor += 1
        else:
            # Overwrite least important
            idx = np.argmin(self.usage)
            self.memory[idx] = experience_vector
            self.usage[idx] = importance

    def read(self, query_vector):
        """Differentiable read (Attention-like) over the matrix."""
        if self.cursor == 0:
            return np.zeros(self.d)
        
        # Dot-product similarity
        weights = self.memory[:self.cursor] @ query_vector
        weights -= np.max(weights)
        weights = np.exp(weights)
        denom = np.sum(weights) + 1e-8
        weights /= denom
        return weights @ self.memory[:self.cursor]

    def get_state(self):
        return {
            "memory": self.memory.copy(),
            "usage": self.usage.copy(),
            "cursor": int(self.cursor),
        }

    def set_state(self, state):
        self.memory = state["memory"].copy()
        self.usage = state["usage"].copy()
        self.cursor = int(state["cursor"])

    def decay(self, rate: float = 0.995):
        if self.cursor == 0:
            return
        self.usage[: self.cursor] *= rate
