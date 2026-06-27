import torch
import torch.nn.functional as F
import math

class EPBlock:
    def __init__(self, d_s, d_r_total, clip_alpha, n_classes, device):
        self.d_s = d_s
        self.d_r_total = d_r_total
        self.clip_alpha = clip_alpha
        self.device = device
        std_w = math.sqrt(2.0 / (d_s + d_r_total))
        self.W = torch.randn(d_s, d_r_total, device=device) * std_w
        std_skip = math.sqrt(2.0 / (d_s + d_s))
        self.W_skip = torch.randn(d_s, d_s, device=device) * std_skip
        self.local_head = torch.randn(n_classes, d_s, device=device) * 0.1

    def activate(self, s):
        return torch.clamp(s, min=0.0, max=self.clip_alpha)

    def free_phase_analytical(self, r_t, s_prev2):
        # r_t: (B, d_r_total), s_prev2: (B, d_s)
        # W: (d_s, d_r_total), W_skip: (d_s, d_s)
        return r_t @ self.W.T + s_prev2 @ self.W_skip.T  # (B, d_s)

    def nudge_phase(self, s0, r_t, s_prev2, y_local, beta, gamma, N_nudge):
        # s0: (B, d_s), y_local: (B,)
        s = s0.clone()
        for _ in range(N_nudge):
            s = s.detach().requires_grad_(True)
            # Energy per sample, summed over batch
            E = (0.5 * (s ** 2).sum(-1)
                 - (s * (r_t @ self.W.T)).sum(-1)
                 - (s * (s_prev2 @ self.W_skip.T)).sum(-1))
            # Local loss
            local_logits = s @ self.local_head.T  # (B, n_classes)
            y_for_loss = y_local.long()
            if y_for_loss.dim() == 0:
                y_for_loss = y_for_loss.unsqueeze(0)
            y_for_loss = y_for_loss.view(-1)
            if local_logits.dim() == 1:
                local_logits = local_logits.unsqueeze(0)
            loss_local = F.cross_entropy(local_logits, y_for_loss)
            total = E.mean() + beta * loss_local
            grad = torch.autograd.grad(total, s)[0]  # (B, d_s)
            s = (s - gamma * grad).detach()
        return s  # (B, d_s)

    def update(self, s_beta, s0, r_t, s_prev2, eta, beta, M_t,
               importance_W=None, W_anchor=None,
               importance_skip=None, skip_anchor=None,
               lambda_reg=None):
        """
        EP weight update with optional EWC-style regularization.
        importance_W:   (d_r_total,) tensor — per-dim importance from CFNR
        W_anchor:       (d_s, d_r_total) — weights to protect (end of prev task)
        lambda_reg:     regularization strength
        """
        if lambda_reg is None:
            lambda_reg = getattr(self, 'lambda_reg', 0.5)
        with torch.no_grad():
            delta = s_beta - s0                          # (B, d_s)
            scale = (eta / beta) * M_t
            dW      = scale * (delta.T @ r_t) / delta.shape[0]
            dW_skip = scale * (delta.T @ s_prev2) / delta.shape[0]

            self.W      += dW
            self.W_skip += dW_skip

            # EWC-style penalty: pull important dims toward anchor
            if importance_W is not None and W_anchor is not None:
                imp = importance_W.unsqueeze(0)          # (1, d_r_total)
                reg_W = lambda_reg * imp * (self.W - W_anchor)
                self.W -= eta * reg_W

            if importance_skip is not None and skip_anchor is not None:
                imp_s = importance_skip.unsqueeze(0)     # (1, d_s)
                reg_s = lambda_reg * imp_s * (self.W_skip - skip_anchor)
                self.W_skip -= eta * reg_s

    def get_output(self, s0):
        return self.activate(s0)
