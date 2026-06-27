import torch

class RecursiveNullSpace:
    """
    Recursive null-space projection tracker for continual learning (CFNR).
    Uses Sherman-Morrison to estimate the covariance (precision) matrix
    of reservoir activations, then projects trainable parameter updates
    onto the null-space of prior tasks at boundaries.
    """
    def __init__(self, d_total, delta, device):
        self.d_total = d_total
        self.device  = device
        self.delta   = delta
        # Precision matrix: starts as (1/delta)*I
        self.P = (1.0 / delta) * torch.eye(d_total, device=device)

    def update(self, r_t):
        """
        Sherman-Morrison rank-1 update.
        r_t: (d_total,) or (B, d_total).
        Performs updates sequentially for each sample in the batch
        to ensure precision matrix is updated for all seen inputs.
        """
        if r_t.dim() == 1:
            r_t = r_t.unsqueeze(0)
        with torch.no_grad():
            for i in range(r_t.shape[0]):
                r = r_t[i]
                Pr    = self.P @ r
                denom = 1.0 + r @ Pr
                self.P = self.P - torch.outer(Pr, Pr) / denom

    def project(self, W_old, W_task):
        """
        Closed-Form Null-Space Projection (Eq. 14).
        Projects the parameter change onto the null space of self.P.
        W_old:  current weights at end of task (d_s, d_total)
        W_task: weights at start of task / anchor (d_s, d_total)
        """
        with torch.no_grad():
            diag_P = torch.diag(self.P)
            diag_inv = 1.0 / (diag_P + 1e-8)
            proj = torch.eye(self.d_total, device=self.device) - self.P * diag_inv.unsqueeze(0)
            W_new = W_old - (W_old - W_task) @ proj.T
        return W_new

    def reset(self, delta):
        self.delta = delta
        self.P = (1.0 / delta) * torch.eye(self.d_total, device=self.device)
