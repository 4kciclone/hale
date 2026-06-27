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
        s0 = self.W @ r_t + self.W_skip @ s_prev2
        return s0

    def nudge_phase(self, s0, r_t, s_prev2, y_local, beta, gamma, N_nudge):
        s = s0.clone().detach().requires_grad_(True)
        for _ in range(N_nudge):
            energy_grad = s - self.W @ r_t - self.W_skip @ s_prev2
            logits = self.local_head @ s
            loss = F.cross_entropy(logits.unsqueeze(0), y_local.view(1).long())
            loss_grad = torch.autograd.grad(loss, s, retain_graph=False)[0]
            with torch.no_grad():
                s_new = s - gamma * (energy_grad + beta * loss_grad)
            s = s_new.detach().requires_grad_(True)
        return s.detach()

    def update(self, s_beta, s0, r_t, s_prev2, eta, beta, M_t):
        with torch.no_grad():
            diff = s_beta - s0
            scale = (eta / beta) * M_t
            self.W += scale * torch.outer(diff, r_t)
            self.W_skip += scale * torch.outer(diff, s_prev2)

    def get_output(self, s0):
        return self.activate(s0)
