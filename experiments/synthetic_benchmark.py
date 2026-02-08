import numpy as np
from scipy.linalg import svd, norm

class OMEGA_ACP:
    def __init__(self, d_model, k_max=20, decay=0.99):
        self.d = d_model
        self.k_max = k_max
        self.l = decay
        self.Q = np.random.randn(d_model, 1)
        self.Q /= (norm(self.Q) + 1e-8)
        self.k = 1
        self.A = np.eye(d_model) * 0.1
        self.P = np.eye(d_model) * 0.1 

    def predict(self, x):
        return (self.A @ x.reshape(-1, 1)).flatten()

    def update(self, x_t, x_next):
        x_t = x_t.reshape(-1, 1); x_next = x_next.reshape(-1, 1)
        pred = self.A @ x_t; err = x_next - pred
        # RLS with clipping
        gain = (self.P @ x_t) / (1.0 + x_t.T @ self.P @ x_t + 1e-6)
        self.A += 0.5 * err @ gain.T
        self.P = (self.P - gain @ x_t.T @ self.P) / 0.999
        # Spectral projection (Crucial for stability)
        u, s, vh = svd(self.A); s = np.clip(s, 0, 0.95); self.A = (u * s) @ vh
        # Arnoldi update
        w = self.A @ self.Q[:, -1]
        for i in range(self.k):
            h = (self.l ** (self.k - i)) * np.dot(w, self.Q[:, i])
            w -= h * self.Q[:, i]
        h_next = norm(w)
        if h_next > 1e-4 and self.k < self.k_max:
            self.Q = np.column_stack([self.Q, w / h_next]); self.k += 1
        elif self.k >= self.k_max:
            u, _, _ = svd(self.Q); self.Q = u[:, :self.k_max // 2]; self.k = self.Q.shape[1]

class RobustLSTM:
    def __init__(self, d_model):
        self.d = d_model; self.h = np.zeros(d_model)
        self.W = np.random.randn(d_model, d_model * 2) * 0.01
    def predict(self, x):
        concat = np.concatenate([x.flatten(), self.h.flatten()])
        return (self.W @ concat)[:self.d]
    def update(self, x_t, x_next):
        concat = np.concatenate([x_t.flatten(), self.h.flatten()])
        pred = self.W @ concat; err = x_next.flatten() - pred[:self.d]
        grad = np.outer(np.clip(err, -1, 1), concat)
        self.W += 0.01 * grad # Backprop-like update
        self.h = np.tanh(pred[:self.d])

def lorenz_step(state, sigma, rho, beta, dt=0.01):
    x, y, z = state
    dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
    return state + np.array([dx, dy, dz]) * dt

def generate_lorenz_data(steps, d_model):
    data = []; projection = np.random.randn(d_model, 3) * 0.5
    state = np.array([1.0, 1.0, 1.0])
    for t in range(steps):
        p = (10.0, 28.0, 8/3) if (t < steps//3 or t > 2*steps//3) else (10.0, 15.0, 2.0)
        state = lorenz_step(state, *p)
        obs = projection @ state + np.random.randn(d_model) * 0.01
        data.append(obs)
    return np.array(data)

def run_benchmark():
    D = 16; STEPS = 600; data = generate_lorenz_data(STEPS, D)
    # Normalize data
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
    models = {"LSTM-Lite": RobustLSTM(D), "OMEGA-ACP": OMEGA_ACP(D)}
    results = {name: [] for name in models}
    print(f"{'Step':<6} | {'Regime':<8} | {'LSTM Err':<10} | {'OMEGA Err':<10}")
    print("-" * 50)
    for t in range(STEPS - 1):
        x_t, x_next = data[t], data[t+1]
        regime = "A1" if t < STEPS//3 else ("B" if t < 2*STEPS//3 else "A2")
        for name, model in models.items():
            pred = model.predict(x_t); err = norm(x_next - pred); results[name].append(err); model.update(x_t, x_next)
        if t % 100 == 0:
            print(f"{t:<6} | {regime:<8} | {results['LSTM-Lite'][-1]:.4f}   | {results['OMEGA-ACP'][-1]:.4f}")
    print("-" * 50 + "\nRESULTS")
    for n in results:
        s3 = STEPS // 3; a1, b, a2 = np.mean(results[n][:s3]), np.mean(results[n][s3:2*s3]), np.mean(results[n][2*s3:])
        forget = (a2 - a1) / a1
        print(f"-> {n:12}: A1: {a1:.4f} | B (Shift): {b:.4f} | A2 (Return): {a2:.4f} | Forget: {forget:+.2%}")

if __name__ == "__main__":
    run_benchmark()
