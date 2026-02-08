import numpy as np


class RegimeDetector:
    """
    Bayesian Entropy Monitor for Out-of-Distribution (OOD) detection.
    Extends detection with Spectral Checksum Structural Identity (SCSI) monitoring.
    """

    def __init__(
        self,
        d_latents,
        window_size=50,
        threshold=2.5,
        scsi_angle_threshold=np.deg2rad(80.0),
        scsi_eigen_threshold=0.8
    ):
        self.d = d_latents
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        self.baseline_mu = None
        self.baseline_sigma = None

        # SCSI parameters
        self.scsi_angle_threshold = scsi_angle_threshold
        self.scsi_eigen_threshold = scsi_eigen_threshold
        self.scsi_reference = None
        self.last_scsi_metrics = {"angle": 0.0, "eig_drift": 0.0}

    def update(self, latent_vector):
        """Monitors statistics of latent activations."""
        self.history.append(latent_vector)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < self.window_size:
            return False  # Still warming up

        current_mu = np.mean(self.history, axis=0)
        current_sigma = np.std(self.history, axis=0)

        if self.baseline_mu is None:
            self.baseline_mu = current_mu
            self.baseline_sigma = current_sigma
            return False

        # Calculate Mahalanobis-like distance / KL Divergence heuristic
        z_score = np.abs(current_mu - self.baseline_mu) / (self.baseline_sigma + 1e-6)
        max_deviation = np.max(z_score)

        if max_deviation > self.threshold:
            # Regime shift detected
            return True

        return False

    def reset_baseline(self):
        """Update baseline after a confirmed regime shift."""
        self.baseline_mu = np.mean(self.history, axis=0)
        self.baseline_sigma = np.std(self.history, axis=0)

    def mark_scsi_baseline(self, scsi_signature, basis_matrix):
        """Stores the current structural signature as the stable reference."""
        if scsi_signature is None and basis_matrix is None:
            return

        angles = self._extract_angles(None if scsi_signature is None else scsi_signature.get("principal_angles"))
        eigenvalues = self._extract_eigenvalues(None if scsi_signature is None else scsi_signature.get("eigenvalues"))
        basis = None
        if basis_matrix is not None:
            basis = np.array(basis_matrix, copy=True)
        self.scsi_reference = {"angles": angles, "eigenvalues": eigenvalues, "basis": basis}

    def check_scsi(self, scsi_signature, basis_matrix, regime_shift=False):
        """
        Evaluates structural identity drift. Returns True when a structural anomaly
        is detected without an accompanying regime shift.
        """
        if scsi_signature is None and basis_matrix is None:
            return False

        current_angles = self._extract_angles(None if scsi_signature is None else scsi_signature.get("principal_angles"))
        current_eigs = self._extract_eigenvalues(None if scsi_signature is None else scsi_signature.get("eigenvalues"))
        current_basis = None if basis_matrix is None else np.array(basis_matrix, copy=False)

        if current_angles.size == 0 and current_eigs.size == 0 and (current_basis is None or current_basis.size == 0):
            return False

        if self.scsi_reference is None or regime_shift:
            baseline_basis = None if current_basis is None else np.array(current_basis, copy=True)
            self.scsi_reference = {"angles": current_angles, "eigenvalues": current_eigs, "basis": baseline_basis}
            return False

        baseline_basis = self.scsi_reference.get("basis")
        if current_basis is not None and baseline_basis is not None and current_basis.size and baseline_basis.size:
            min_dim = min(baseline_basis.shape[1], current_basis.shape[1])
            if min_dim > 0:
                cross = baseline_basis[:, :min_dim].T @ current_basis[:, :min_dim]
                _, sing_vals, _ = np.linalg.svd(cross, full_matrices=False)
                sing_vals = np.clip(sing_vals, -1.0, 1.0)
                principal_angles = np.arccos(sing_vals)
                max_angle = float(np.max(principal_angles)) if principal_angles.size else 0.0
            else:
                max_angle = 0.0
        else:
            max_angle = float(np.max(current_angles)) if current_angles.size else 0.0

        ref_eigs = self.scsi_reference["eigenvalues"]
        if ref_eigs.size and current_eigs.size:
            cur_sorted = np.sort_complex(current_eigs)
            ref_sorted = np.sort_complex(ref_eigs)
            max_len = max(cur_sorted.size, ref_sorted.size)
            cur_pad = np.pad(cur_sorted, (0, max_len - cur_sorted.size), constant_values=0)
            ref_pad = np.pad(ref_sorted, (0, max_len - ref_sorted.size), constant_values=0)
            diff = np.linalg.norm(cur_pad - ref_pad)
            base = max(np.linalg.norm(ref_pad), 1.0)
            eig_drift = diff / base
        else:
            eig_drift = 0.0

        anomaly = (max_angle > self.scsi_angle_threshold) and (eig_drift > self.scsi_eigen_threshold)
        self.last_scsi_metrics = {"angle": max_angle, "eig_drift": eig_drift}
        return anomaly

    @staticmethod
    def _extract_angles(values):
        if values is None:
            return np.array([])
        arr = np.array(values, dtype=float)
        return arr

    @staticmethod
    def _extract_eigenvalues(values):
        if values is None:
            return np.array([], dtype=complex)
        return np.array(values, dtype=complex)

    def get_state(self):
        return {
            "history": np.array(self.history, copy=True),
            "baseline_mu": None if self.baseline_mu is None else self.baseline_mu.copy(),
            "baseline_sigma": None if self.baseline_sigma is None else self.baseline_sigma.copy(),
            "scsi_reference": None
            if self.scsi_reference is None
            else {
                "angles": None
                if self.scsi_reference.get("angles") is None
                else np.array(self.scsi_reference["angles"], copy=True),
                "eigenvalues": None
                if self.scsi_reference.get("eigenvalues") is None
                else np.array(self.scsi_reference["eigenvalues"], copy=True),
                "basis": None
                if self.scsi_reference.get("basis") is None
                else np.array(self.scsi_reference["basis"], copy=True),
            },
            "last_scsi_metrics": dict(self.last_scsi_metrics),
        }

    def set_state(self, state):
        history = state.get("history")
        self.history = [] if history is None else [h.copy() for h in history]
        self.baseline_mu = None
        self.baseline_sigma = None
        if state.get("baseline_mu") is not None:
            self.baseline_mu = state["baseline_mu"].copy()
        if state.get("baseline_sigma") is not None:
            self.baseline_sigma = state["baseline_sigma"].copy()

        ref = state.get("scsi_reference")
        if ref is None:
            self.scsi_reference = None
        else:
            self.scsi_reference = {
                "angles": None if ref.get("angles") is None else np.array(ref["angles"], copy=True),
                "eigenvalues": None if ref.get("eigenvalues") is None else np.array(ref["eigenvalues"], copy=True),
                "basis": None if ref.get("basis") is None else np.array(ref["basis"], copy=True),
            }
        self.last_scsi_metrics = dict(state.get("last_scsi_metrics", {"angle": 0.0, "eig_drift": 0.0}))
