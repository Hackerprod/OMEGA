import numpy as np
from scipy.linalg import svd, norm, qr

from .kernels import arnoldi_iteration

class ACPModule:
    """
    Arnoldi-Causal Projection (ACP) Module.
    Refines predictions by projecting them onto the causal subspace.
    """
    def __init__(
        self,
        d_model,
        k_max=16,
        decay_lambda=0.99,
        tau=1e-4,
        rls_lambda=0.99,
        alpha=1e-3,
        frobenius_radius=1.0,
        orthogonality_tol=1e-4,
        svd_interval=5,
        adaptive_basis=True,
        basis_budget_ratio=0.5,
        recycle_ratio=0.6,
        min_basis_size=4,
    ):
        self.d = d_model
        self.k_max = max(1, int(k_max))
        self.l = decay_lambda
        self.tau = tau
        self.rls_lambda = rls_lambda
        self.alpha = alpha
        self.frobenius_radius = frobenius_radius
        self.orthogonality_tol = orthogonality_tol
        self.svd_interval = max(1, int(svd_interval))
        self._base_svd_interval = self.svd_interval
        self.adaptive_basis = bool(adaptive_basis)
        self.basis_budget_ratio = float(np.clip(basis_budget_ratio, 0.1, 1.0))
        self.recycle_ratio = float(np.clip(recycle_ratio, 0.1, 0.95))
        self.min_basis_size = max(1, int(min_basis_size))
        self._compress_rank_limit = None

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
        self._calibrate_basis_limits(reallocate=True)

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
        w_vector = (self.A @ q_last).flatten()
        residual, h_column, gradients = arnoldi_iteration(self.Q, w_vector, self.l, self.k)
        if h_column.shape[0] >= self.k:
            self.H[:self.k, self.k - 1] = h_column[:self.k]
        h_next = float(h_column[-1]) if h_column.size else 0.0
        self.H[self.k, self.k - 1] = h_next
        w = residual.reshape(-1, 1)
        monotonic_gradients = gradients.tolist()

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

        if self.k >= self.k_max:
            self._conditional_restart(h_next)

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
        target_cap = self._compress_rank_limit if self._compress_rank_limit is not None else self.k_max // 2
        target_dim = max(1, min(target_cap, vt.shape[0], self.k_max))
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
            "prev_basis": None if self.prev_basis is None else self.prev_basis.copy(),
            "adaptive_basis": bool(self.adaptive_basis),
            "basis_budget_ratio": float(self.basis_budget_ratio),
            "recycle_ratio": float(self.recycle_ratio),
            "min_basis_size": int(self.min_basis_size),
            "k_max": int(self.k_max),
            "svd_interval": int(self.svd_interval)
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
        self.adaptive_basis = bool(state.get("adaptive_basis", self.adaptive_basis))
        self.basis_budget_ratio = float(state.get("basis_budget_ratio", self.basis_budget_ratio))
        self.recycle_ratio = float(state.get("recycle_ratio", self.recycle_ratio))
        self.min_basis_size = int(state.get("min_basis_size", self.min_basis_size))
        restored_k_max = int(state.get("k_max", self.k_max))
        if restored_k_max != self.k_max:
            self.k_max = max(1, restored_k_max)
            self.H = np.zeros((self.k_max + 1, self.k_max))
        self.svd_interval = int(state.get("svd_interval", self.svd_interval))
        self._base_svd_interval = self.svd_interval
        self._calibrate_basis_limits(reallocate=True)

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

    def _calibrate_basis_limits(self, reallocate=False):
        """Adapt k_max and svd_interval to the current dimensionality budget."""
        if not self.adaptive_basis:
            self.k_max = max(1, min(self.k_max, self.d))
            if reallocate:
                self.H = np.zeros((self.k_max + 1, self.k_max))
            self._compress_rank_limit = max(1, min(self.k_max, self.k_max // 2 or 1))
            return

        budget = int(self.d * self.basis_budget_ratio)
        budget = max(self.min_basis_size, min(self.k_max, budget))
        budget = max(1, min(budget, self.d))
        needs_realloc = (budget != self.k_max) or (self.H.shape[1] != budget)
        self.k_max = budget
        self._compress_rank_limit = max(self.min_basis_size, int(self.k_max * 0.75))
        self._compress_rank_limit = min(self._compress_rank_limit, self.k_max)

        if needs_realloc or reallocate:
            new_h = np.zeros((self.k_max + 1, self.k_max))
            rows = min(new_h.shape[0], self.H.shape[0])
            cols = min(new_h.shape[1], self.H.shape[1])
            new_h[:rows, :cols] = self.H[:rows, :cols]
            self.H = new_h

        if self.Q.shape[1] > self.k_max:
            self.Q = self.Q[:, :self.k_max]
        self.k = min(self.k, self.Q.shape[1])
        if self.prev_basis is not None and self.prev_basis.shape[1] > self.k_max:
            self.prev_basis = self.prev_basis[:, :self.k_max]

        adaptive_interval = max(1, self.k_max // 3)
        base_interval = self._base_svd_interval
        self.svd_interval = max(1, min(base_interval, adaptive_interval if adaptive_interval > 0 else base_interval))

    def _conditional_restart(self, h_next: float):
        """Recycle part of the Krylov basis when growth stalls at the budget frontier."""
        if not self.adaptive_basis:
            return
        if self.k < self.k_max:
            return
        if h_next > self.tau * 2.0:
            return
        target_dim = max(self.min_basis_size, int(self.k_max * self.recycle_ratio))
        target_dim = min(target_dim, self.k_max)
        if target_dim >= self.k:
            return
        self.Q = self.Q[:, :target_dim]
        self.k = target_dim
        self.H = np.zeros((self.k_max + 1, self.k_max))
        self.prev_basis = self.Q.copy()
