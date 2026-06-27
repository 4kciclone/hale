import torch

class RecursiveNullSpace:
    def __init__(self, d_total, delta, device):
        self.d_total = d_total
        self.device = device
        self.P = (1.0 / delta) * torch.eye(d_total, device=device)

    def update(self, r_t):
        with torch.no_grad():
            Pr = self.P @ r_t
            denom = 1.0 + r_t @ Pr
            self.P = self.P - torch.outer(Pr, Pr) / denom

    def project(self, W_old, W_task):
        with torch.no_grad():
            diag_P = torch.diag(self.P)
            diag_inv = 1.0 / (diag_P + 1e-8)
            proj = torch.eye(self.d_total, device=self.device) - self.P * diag_inv.unsqueeze(0)
            W_new = W_old - (W_old - W_task) @ proj.T
        return W_new

    def reset(self, delta):
        self.P = (1.0 / delta) * torch.eye(self.d_total, device=self.device)
