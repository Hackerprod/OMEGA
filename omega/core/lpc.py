import numpy as np


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


class LocalPredictiveUnit:
    """
    OMEGA v3: Difference Target Propagation (DTP).
    Each unit also maintains a local causal operator learned via RLS against the global ACP basis.
    """

    def __init__(self, d_in, d_out, rls_lambda=0.99, alpha=1e-3, frobenius_radius=1.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rls_lambda = rls_lambda
        self.alpha = alpha
        self.frobenius_radius = frobenius_radius

        # Forward weights
        self.W = np.random.randn(d_out, d_in) * (1.0 / np.sqrt(d_in))
        # Inverse/Backward weights (Feedback)
        self.V = np.random.randn(d_in, d_out) * (1.0 / np.sqrt(d_out))
        # RLS state for Forward
        self.P = np.eye(d_in)

        # Local causal operator (A_L) and covariance
        self.A_local = np.eye(d_out, d_in)
        self.P_local = np.eye(d_in) / max(self.alpha, 1e-6)

        self.last_x = None
        self.last_z_pre = None  # Pre-activation
        self.last_z = None  # Post-activation

    def forward(self, x):
        self.last_x = x.reshape(-1, 1)
        self.last_z_pre = self.W @ self.last_x
        self.last_z = np.tanh(self.last_z_pre)
        return self.last_z.flatten()

    def propagate_target(self, target_z):
        """
        Implements Difference Target Propagation (DTP).
        Target_x = last_x + V(target_z) - V(last_z)
        This ensures the target respects the local non-linear mapping.
        """
        target_z = target_z.reshape(-1, 1)
        # Compute the target for the layer below
        delta_x = self.V @ target_z - self.V @ self.last_z
        target_x = self.last_x + delta_x
        return target_x.flatten()

    def local_update(self, target_z, basis_vector=None):
        """
        Updates Forward weights (W), Backward weights (V), and the local causal operator A_L.
        """
        target_z = target_z.reshape(-1, 1)

        # 1. Update Forward W (using RLS on the linear part to reach atanh(target_z))
        target_z_clipped = np.clip(target_z, -0.99, 0.99)
        z_pre_target = np.arctanh(target_z_clipped)

        error_f = z_pre_target - self.last_z_pre
        gain = (self.P @ self.last_x) / (1.0 + self.last_x.T @ self.P @ self.last_x + 1e-8)
        self.W += error_f @ gain.T
        self.P = (self.P - gain @ self.last_x.T @ self.P) / self.rls_lambda
        self.P = 0.5 * (self.P + self.P.T)

        # 2. Update Inverse V (The layer learns to invert its own forward pass)
        error_v = self.last_x - self.V @ self.last_z
        self.V += 0.1 * error_v @ self.last_z.T

        # 3. Update Local Causal Operator using the ACP basis
        if basis_vector is not None:
            phi = basis_vector.reshape(-1, 1)
            denom = float(self.rls_lambda + phi.T @ self.P_local @ phi)
            k_gain = (self.P_local @ phi) / (denom + 1e-8)
            err_local = target_z - self.A_local @ phi
            self.A_local += err_local @ k_gain.T

            self.P_local = (self.P_local - k_gain @ phi.T @ self.P_local) / self.rls_lambda
            self.P_local += self.alpha * np.eye(self.d_in)
            self.P_local = 0.5 * (self.P_local + self.P_local.T)

            # Regularize A_L
            self.A_local *= (1.0 - self.alpha)
            frob = np.linalg.norm(self.A_local, ord="fro")
            if frob > self.frobenius_radius:
                self.A_local *= self.frobenius_radius / (frob + 1e-8)

            if self.d_in == self.d_out:
                eigvals = np.linalg.eigvals(self.A_local)
                rho = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
                if rho >= 1.0:
                    self.A_local *= (0.99 / (rho + 1e-8))

        # Recalculate local state for metrics
        new_z = np.tanh(self.W @ self.last_x)
        return np.linalg.norm(target_z - new_z)

    def project_basis(self, basis_vector):
        """
        Projects the global ACP basis through the local operator to produce a basis estimate for lower layers.
        """
        if basis_vector is None:
            return None
        phi = basis_vector.reshape(-1, 1)
        projected = self.A_local @ phi
        norm_proj = np.linalg.norm(projected)
        if norm_proj < 1e-8:
            return basis_vector
        return (projected / norm_proj).flatten()

    def get_state(self):
        """Create a snapshot of the local predictive unit."""
        return {
            "W": self.W.copy(),
            "V": self.V.copy(),
            "P": self.P.copy(),
            "A_local": self.A_local.copy(),
            "P_local": self.P_local.copy()
        }

    def set_state(self, state):
        """Restore parameters from a snapshot created by get_state."""
        self.W = state["W"].copy()
        self.V = state["V"].copy()
        self.P = state["P"].copy()
        self.A_local = state["A_local"].copy()
        self.P_local = state["P_local"].copy()
