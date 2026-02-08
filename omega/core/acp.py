import numpy as np
from scipy.linalg import svd, norm, qr

class ACPModule:
    """
    Arnoldi-Causal Projection (ACP) Module.
    Refines predictions by projecting them onto the causal subspace.
    """
    def __init__(self, d_model, k_max=16, decay_lambda=0.99, tau=1e-4,
                 rls_lambda=0.99, alpha=1e-3, frobenius_radius=1.0,
                 orthogonality_tol=1e-4, svd_interval=5):
        self.d = d_model
        self.k_max = k_max
        self.l = decay_lambda
        self.tau = tau
        self.rls_lambda = rls_lambda
        self.alpha = alpha
        self.frobenius_radius = frobenius_radius
        self.orthogonality_tol = orthogonality_tol
        self.svd_interval = max(1, int(svd_interval))
        self.Q = np.random.randn(d_model, 1); self.Q /= (norm(self.Q) + 1e-8)
        self.k = 1
        self.A = np.eye(d_model) * 0.1
        self.P = np.eye(d_model) / max(self.alpha, 1e-6)
        self.H = np.zeros((self.k_max + 1, self.k_max))
        self.prev_basis = None
        self.scsi_signature = {"eigenvalues": None, "principal_angles": None}
        self.last_orth_error = 0.0
        self.last_monotonic_gradient = []
        self.step_counter = 0
        self.power_vector = np.random.randn(d_model, 1)
        self.power_vector /= (norm(self.power_vector) + 1e-8)
        self._spectral_radius = 0.0

    def refine_prediction(self, raw_pred):
        """
        Projects the raw neural prediction onto the Causal Subspace.
        This filters out components that don't match the learned system dynamics.
        """
        raw_pred = raw_pred.reshape(-1, 1)
        # Project onto Q basis: Q Q^T z
        refined = self.Q @ (self.Q.T @ raw_pred)
        return refined.flatten()

    def update_operator(self, x_t, x_next):
        x_t, x_next = x_t.reshape(-1, 1), x_next.reshape(-1, 1)

        if self.k > 0:
            q_t = self.Q[:, self.k - 1].reshape(-1, 1)
        else:
            norm_x = norm(x_t)
            q_t = x_t / (norm_x + 1e-8)

        phi = q_t
        denom = float(self.rls_lambda + phi.T @ self.P @ phi)
        gain = (self.P @ phi) / (denom + 1e-8)
        error = x_next - self.A @ phi
        self.A += error @ gain.T

        self.P = (self.P - gain @ phi.T @ self.P) / self.rls_lambda
        self.P += self.alpha * np.eye(self.d)
        self.P = 0.5 * (self.P + self.P.T)

        self.A *= (1.0 - self.alpha)
        frob = norm(self.A, ord='fro')
        if frob > self.frobenius_radius:
            self.A *= self.frobenius_radius / (frob + 1e-8)

        v_new = self.A @ self.power_vector
        radius_est = norm(v_new)
        if radius_est > 0:
            self.power_vector = v_new / (radius_est + 1e-8)
        self._spectral_radius = float(radius_est)

        if self._spectral_radius >= 1.0:
            self.A *= (0.99 / (self._spectral_radius + 1e-8))
            self._spectral_radius = 0.99
            self.power_vector = self.A @ self.power_vector
            pv_norm = norm(self.power_vector)
            if pv_norm > 0:
                self.power_vector /= pv_norm

    def step(self, seed_vector=None):
        """
        Updates the causal Krylov basis using the latest transition operator.
        Optionally seeds the base vector with the provided observation.
        """
        self.step_counter += 1
        prev_basis = self.Q[:, :self.k].copy() if self.k > 0 else None
        spectral_signature = None

        if seed_vector is not None:
            v = seed_vector.reshape(-1, 1)
            v_norm = norm(v)
            if v_norm > 1e-8:
                self.Q[:, 0:1] = v / v_norm
                self.k = max(self.k, 1)

        q_last = self.Q[:, self.k - 1].reshape(-1, 1)
        w = self.A @ q_last
        monotonic_gradients = []
        for i in range(self.k):
            q_i = self.Q[:, i].reshape(-1, 1)
            dot_val = float(q_i.T @ w)
            weight = self.l ** (self.k - 1 - i)
            h_mag = weight * abs(dot_val)
            coeff = np.sign(dot_val) * h_mag
            self.H[i, self.k - 1] = coeff
            w -= coeff * q_i
            exponent = self.k - 1 - i
            if exponent > 0:
                grad_mag = exponent * (self.l ** (exponent - 1)) * abs(dot_val)
            else:
                grad_mag = 0.0
            monotonic_gradients.append(grad_mag)

        h_next = float(norm(w))
        self.H[self.k, self.k - 1] = h_next

        if h_next > self.tau and self.k < self.k_max:
            q_new = (w / h_next).flatten()
            self.Q = np.column_stack([self.Q, q_new])
            self.k += 1
            self.H[:, self.k - 1] = 0.0
        elif self.k >= self.k_max:
            if self.step_counter % self.svd_interval == 0:
                spectral_signature = self._compress_basis()
            else:
                monotonic_gradients = [0.0] * self.k
        else:
            self.H[self.k, self.k - 1] = 0.0

        self._enforce_orthogonality()
        self.last_monotonic_gradient = monotonic_gradients
        self._update_scsi(prev_basis, spectral_signature)
        return self.Q

    def _compress_basis(self):
        """Spectral compression via truncated SVD of the Hessenberg matrix."""
        active_cols = min(self.k, self.H.shape[1])
        if active_cols == 0:
            return None

        H_k = self.H[:active_cols + 1, :active_cols]
        u, s, vt = svd(H_k, full_matrices=False)
        target_dim = max(1, min(self.k_max // 2, vt.shape[0]))
        V_r = vt[:target_dim, :].T  # shape (active_cols, target_dim)
        compressed_basis = self.Q[:, :active_cols] @ V_r
        q_new, _ = qr(compressed_basis, mode='economic')
        self.Q = q_new[:, :target_dim]
        self.k = self.Q.shape[1]
        spectral_signature = np.linalg.eigvals(H_k[:target_dim, :target_dim])
        self.H = np.zeros((self.k_max + 1, self.k_max))
        return spectral_signature

    def _update_scsi(self, prev_basis, spectral_signature=None):
        """Updates the spectral checksum structural identity (SCSI) metrics."""
        current_basis = self.Q[:, :self.k]

        if prev_basis is not None and prev_basis.size > 0 and current_basis.size > 0:
            shared_dim = min(prev_basis.shape[1], current_basis.shape[1])
            if shared_dim > 0:
                cross = prev_basis[:, :shared_dim].T @ current_basis[:, :shared_dim]
                _, sing_vals, _ = svd(cross, full_matrices=False)
                sing_vals = np.clip(sing_vals, -1.0, 1.0)
                principal_angles = np.arccos(sing_vals)
            else:
                principal_angles = np.array([])
        else:
            principal_angles = np.array([])

        if spectral_signature is None and self.k > 0:
            H_active = self.H[:self.k, :self.k]
            spectral_signature = np.linalg.eigvals(H_active) if H_active.size > 0 else np.array([])
        elif spectral_signature is None:
            spectral_signature = np.array([])

        self.scsi_signature = {
            "eigenvalues": spectral_signature,
            "principal_angles": principal_angles
        }
        self.prev_basis = current_basis.copy() if current_basis.size > 0 else None

    def get_state(self):
        """Returns a deep copy of the internal ACP state."""
        return {
            "A": self.A.copy(),
            "Q": self.Q.copy(),
            "H": self.H.copy(),
            "P": self.P.copy(),
            "k": int(self.k),
            "last_orth_error": float(self.last_orth_error),
            "last_monotonic_gradient": np.array(self.last_monotonic_gradient, copy=True),
            "step_counter": int(self.step_counter),
            "power_vector": self.power_vector.copy(),
            "spectral_radius": float(self._spectral_radius),
            "scsi": {
                "eigenvalues": None if self.scsi_signature["eigenvalues"] is None
                else np.array(self.scsi_signature["eigenvalues"], copy=True),
                "principal_angles": None if self.scsi_signature["principal_angles"] is None
                else np.array(self.scsi_signature["principal_angles"], copy=True)
            },
            "prev_basis": None if self.prev_basis is None else self.prev_basis.copy()
        }

    def set_state(self, state):
        """Restores the ACP state from a snapshot produced by get_state."""
        self.A = state["A"].copy()
        self.Q = state["Q"].copy()
        self.H = state["H"].copy()
        self.P = state["P"].copy()
        self.k = int(state["k"])
        self.last_orth_error = float(state.get("last_orth_error", 0.0))
        grad = state.get("last_monotonic_gradient")
        self.last_monotonic_gradient = [] if grad is None else list(np.array(grad, copy=True))
        self.step_counter = int(state.get("step_counter", self.step_counter))
        self.power_vector = state.get("power_vector", self.power_vector).copy()
        self._spectral_radius = float(state.get("spectral_radius", self._spectral_radius))

        scsi_state = state.get("scsi", {})
        eigs = scsi_state.get("eigenvalues")
        angles = scsi_state.get("principal_angles")
        self.scsi_signature = {
            "eigenvalues": None if eigs is None else np.array(eigs, copy=True),
            "principal_angles": None if angles is None else np.array(angles, copy=True)
        }
        prev_basis = state.get("prev_basis")
        self.prev_basis = None if prev_basis is None else prev_basis.copy()

    def _enforce_orthogonality(self):
        """Re-orthogonalises Q when numerical drift exceeds the tolerance."""
        if self.Q.size == 0:
            return
        gram = self.Q.T @ self.Q
        deviation = gram - np.eye(self.Q.shape[1])
        deviation_norm = norm(deviation, ord='fro')
        self.last_orth_error = deviation_norm
        if deviation_norm > self.orthogonality_tol:
            q_new, _ = qr(self.Q, mode='economic')
            self.Q = q_new[:, :self.Q.shape[1]]

    @property
    def spectral_radius(self):
        return float(self._spectral_radius)
