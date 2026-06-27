import torch

def to_scalar_label(y, device):
    if not isinstance(y, torch.Tensor):
        return torch.tensor(y, dtype=torch.long, device=device)
    y = y.squeeze()
    if y.dim() == 0:
        return y.long().to(device)
    return y.view(-1)[0].long().to(device)

class ClassicESN:
    def __init__(self, d_in, d_r=512, n_classes=2, delta=1.0, device='cpu'):
        self.device = device
        self.d_in = d_in
        self.d_r = d_r
        self.n_classes = n_classes
        self.W_in = torch.randn(d_r, d_in, device=device) * 0.1
        self.W_R = self._init_reservoir(d_r, rho=0.9, device=device)
        self.r = torch.zeros(d_r, device=device)
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
        self.r = 0.5 * self.r + 0.5 * torch.tanh(self.W_in @ x_t + self.W_R @ self.r)
        return self.r

    def reset(self):
        self.r = torch.zeros(self.d_r, device=self.device)

    def train_sequence(self, x_seq, y):
        x_seq = x_seq.to(self.device)
        y = to_scalar_label(y, self.device)
        self.reset()
        for t in range(x_seq.shape[0]):
            r_t = self.step(x_seq[t])
        with torch.no_grad():
            Pr = self.P_rls @ r_t
            denom = 1.0 + r_t @ Pr
            k = Pr / denom
            y_onehot = torch.zeros(self.n_classes, device=self.device)
            y_onehot[y] = 1.0
            pred = self.W_out @ r_t
            error = y_onehot - pred
            self.W_out = self.W_out + torch.outer(error, k)
            self.P_rls = self.P_rls - torch.outer(k, Pr)
        return (pred.argmax() == y).item()

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        for x_seq, y in dataloader:
            x_seq = x_seq.squeeze(0).to(self.device)
            y = to_scalar_label(y, self.device)
            self.reset()
            with torch.no_grad():
                for t in range(x_seq.shape[0]):
                    r_t = self.step(x_seq[t])
                pred = self.W_out @ r_t
                if pred.argmax() == y.item():
                    correct += 1
            total += 1
        return correct / total if total > 0 else 0.0
