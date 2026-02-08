import numpy as np

class SymbolicInterface:
    """
    Adaptive Neuro-Symbolic Bridge.
    Learns symbolic anchors from the data distribution via online competitive learning.
    Enforces a Lipschitz constraint on the anchor updates to keep the mapping contractive.
    """

    def __init__(self, d_model, n_predicates, anchor_radius=5.0, max_update_norm=0.1):
        self.d = d_model
        self.n = n_predicates
        self.anchor_radius = anchor_radius
        self.max_update_norm = max_update_norm
        # Learnable anchors (Prototypes)
        self.anchors = np.random.randn(n_predicates, d_model)
        self._project_anchors()
        self.usage_counts = np.ones(n_predicates)
        self.last_update_norm = 0.0

    def map_and_learn(self, z, lr=0.05):
        """
        Maps vector to symbol and updates the anchor to track the data cluster.
        The update is clipped to satisfy a Lipschitz bound.
        """
        # Distances to prototypes
        diffs = self.anchors - z
        dist_sq = np.sum(diffs**2, axis=1)
        symbol_id = np.argmin(dist_sq)

        # Competitive Learning with Lipschitz-bounded update
        delta = z - self.anchors[symbol_id]
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0.0:
            max_step = min(self.max_update_norm, lr * delta_norm)
            scaled_delta = delta * (max_step / (delta_norm + 1e-8))
        else:
            scaled_delta = np.zeros_like(delta)

        self.anchors[symbol_id] += scaled_delta
        self._project_anchor(symbol_id)
        self.usage_counts[symbol_id] += 1
        self.last_update_norm = np.linalg.norm(scaled_delta)

        confidence = 1.0 / (1.0 + np.sqrt(dist_sq[symbol_id]))
        return symbol_id, confidence

    def _project_anchor(self, idx):
        norm_anchor = np.linalg.norm(self.anchors[idx])
        if norm_anchor > self.anchor_radius:
            self.anchors[idx] *= (self.anchor_radius / (norm_anchor + 1e-8))

    def _project_anchors(self):
        for i in range(self.n):
            self._project_anchor(i)

    def get_state(self):
        return {
            "anchors": self.anchors.copy(),
            "usage_counts": self.usage_counts.copy(),
            "last_update_norm": float(self.last_update_norm),
            "anchor_radius": float(self.anchor_radius),
            "max_update_norm": float(self.max_update_norm),
        }

    def set_state(self, state):
        self.anchors = state["anchors"].copy()
        self.usage_counts = state["usage_counts"].copy()
        self.anchor_radius = float(state.get("anchor_radius", self.anchor_radius))
        self.max_update_norm = float(state.get("max_update_norm", self.max_update_norm))
        self.last_update_norm = float(state.get("last_update_norm", 0.0))

class LogicEngine:
    def __init__(self):
        self.rules = {}

    def add_rule(self, premise, conclusion):
        self.rules[premise] = conclusion

    def infer(self, symbol_id):
        return self.rules.get(symbol_id, None)
