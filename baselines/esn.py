import torch

def to_scalar_label(y, device):
    if not isinstance(y, torch.Tensor):
        return torch.tensor(y, dtype=torch.long, device=device)
    return y.squeeze().long().to(device)

class ClassicESN:
    def __init__(self, d_in, d_r=512, n_classes=2, delta=1.0, device='cpu'):
        self.device = device
        self.d_in = d_in
        self.d_r = d_r
        self.n_classes = n_classes
        self.W_in = torch.randn(d_r, d_in, device=device) * 0.1
        self.W_R = self._init_reservoir(d_r, rho=0.9, device=device)
        self.r = torch.zeros(1, d_r, device=device)
        self.P_rls = (1.0 / delta) * torch.eye(d_r, device=device)
        self.W_out = torch.zeros(n_classes, d_r, device=device)

    def _init_reservoir(self, d_r, rho, device):
        W = torch.randn(d_r, d_r, device=device)
        eigenvalues = torch.linalg.eigvals(W)
        max_eig = eigenvalues.abs().max().item()
        if max_eig > 0:
            W = W * (rho / max_eig)
        return W

    def step(self, x_t):
        # x_t: (B, d_in), self.r: (B, d_r)
        self.r = 0.5 * self.r + 0.5 * torch.tanh(
            x_t @ self.W_in.T + self.r @ self.W_R.T
        )
        return self.r

    def reset(self, B=1):
        self.r = torch.zeros(B, self.d_r, device=self.device)

    def train_sequence(self, x_seq, y):
        x_seq = x_seq.to(self.device)
        y = to_scalar_label(y, self.device)
        B = x_seq.shape[0]
        self.reset(B)
        for t in range(x_seq.shape[1]):
            r_t = self.step(x_seq[:, t, :])
        total_correct = 0
        with torch.no_grad():
            for i in range(B):
                h_i = r_t[i]
                y_i = y[i].item()
                Pr = self.P_rls @ h_i
                denom = 1.0 + h_i @ Pr
                k = Pr / denom
                y_onehot = torch.zeros(self.n_classes, device=self.device)
                y_onehot[y_i] = 1.0
                pred = self.W_out @ h_i
                error = y_onehot - pred
                self.W_out = self.W_out + torch.outer(error, k)
                self.P_rls = self.P_rls - torch.outer(k, Pr)
                if pred.argmax().item() == y_i:
                    total_correct += 1
        return total_correct / B

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        for x_seq, y in dataloader:
            x_seq = x_seq.to(self.device)
            y = to_scalar_label(y, self.device)
            B = x_seq.shape[0]
            self.reset(B)
            with torch.no_grad():
                for t in range(x_seq.shape[1]):
                    r_t = self.step(x_seq[:, t, :])
                pred = r_t @ self.W_out.T
                correct += (pred.argmax(dim=1) == y).sum().item()
            total += B
        return correct / total if total > 0 else 0.0
