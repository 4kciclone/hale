import torch

class RecursiveNullSpace:
    """
    Recursive importance tracker for continual learning.
    Uses Sherman-Morrison to estimate the Fisher information
    of reservoir activations, then applies importance-weighted
    regularization at task boundaries.
    """
    def __init__(self, d_total, delta, device):
        self.d_total = d_total
        self.device  = device
        self.delta   = delta
        # Precision matrix: starts as (1/delta)*I
        # After updates, diagonal encodes activation importance
        self.P = (1.0 / delta) * torch.eye(d_total, device=device)
        # Store importance diagonal extracted at task boundary
        self.importance = torch.zeros(d_total, device=device)

    def update(self, r_t):
        """
        Sherman-Morrison rank-1 update.
        r_t: (d_total,) or (B, d_total) — uses mean for batch.
        """
        if r_t.dim() == 2:
            r = r_t.mean(dim=0)
        else:
            r = r_t
        with torch.no_grad():
            Pr    = self.P @ r
            denom = 1.0 + r @ Pr
            self.P = self.P - torch.outer(Pr, Pr) / denom

    def compute_importance(self):
        """
        Extract per-dimension importance from precision matrix.
        High diagonal value = dimension was frequently activated
        = important for past task = should be protected.
        Importance = initial_P_diag - current_P_diag
        (dimensions that changed most were most used)
        """
        initial_diag = 1.0 / self.delta
        current_diag = torch.diag(self.P)
        # Importance = how much the precision dropped (= how much info absorbed)
        self.importance = torch.clamp(initial_diag - current_diag, min=0.0)
        return self.importance

    def project(self, W_old, W_task):
        """
        Importance-weighted projection at task boundary.
        Dimensions important for past task are pulled toward W_task.
        Dimensions not important are free to change.

        W_old:  current weights (d_s, d_total)
        W_task: weights at end of task (d_s, d_total)
        Returns W_new that preserves important dimensions.
        """
        with torch.no_grad():
            importance = self.compute_importance()

            # Normalize importance to [0, 1]
            imp_max = importance.max()
            if imp_max > 1e-8:
                importance_norm = importance / imp_max
            else:
                # No information absorbed — nothing to protect
                return W_old

            # Protection mask: high importance → pull toward W_task
            # low importance  → leave free (W_old)
            # W_new = W_old - importance_norm * (W_old - W_task)
            # = (1 - imp) * W_old + imp * W_task
            protection = importance_norm.unsqueeze(0)   # (1, d_total)
            W_new = W_old - protection * (W_old - W_task)

        return W_new

    def reset(self, delta):
        self.delta = delta
        self.P = (1.0 / delta) * torch.eye(
            self.d_total, device=self.device)
        self.importance = torch.zeros(self.d_total, device=self.device)
