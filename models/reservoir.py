import torch

class HeterogeneousReservoir:
    def __init__(self, d_in, d_r, alpha_list, rho_list, device):
        self.d_in = d_in
        self.d_r = d_r
        self.J = len(alpha_list)
        self.device = device
        self.alpha_list = alpha_list
        self.rho_list = rho_list
        self.W_in_list = []
        self.W_R_list = []
        self.r_list = []
        for j in range(self.J):
            W_in = torch.randn(d_r, d_in, device=device) * 0.1
            W_R = self._init_W_R(d_r, rho_list[j], device)
            r = torch.zeros(d_r, device=device)
            self.W_in_list.append(W_in)
            self.W_R_list.append(W_R)
            self.r_list.append(r)

    def _init_W_R(self, d_r, rho, device):
        W = torch.randn(d_r, d_r, device=device)
        eigenvalues = torch.linalg.eigvals(W)
        max_eig = eigenvalues.abs().max().item()
        if max_eig > 0:
            W = W * (rho / max_eig)
        return W

    def step(self, h):
        states = []
        for j in range(self.J):
            alpha = self.alpha_list[j]
            pre = self.W_in_list[j] @ h + self.W_R_list[j] @ self.r_list[j]
            self.r_list[j] = (1 - alpha) * self.r_list[j] + alpha * torch.tanh(pre)
            states.append(self.r_list[j])
        return torch.cat(states)

    def reset(self):
        for j in range(self.J):
            self.r_list[j] = torch.zeros(self.d_r, device=self.device)

    def get_state_copies(self):
        return [r.clone() for r in self.r_list]

    def set_states(self, state_copies):
        for j in range(self.J):
            self.r_list[j] = state_copies[j]
