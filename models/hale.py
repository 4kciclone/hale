import torch
import torch.nn.functional as F
import copy
from .ep_block import EPBlock
from .reservoir import HeterogeneousReservoir
from .critic import NeuromdulatoryCtric
from .cfnr import RecursiveNullSpace

def to_scalar_label(y, device):
    if not isinstance(y, torch.Tensor):
        return torch.tensor(y, dtype=torch.long, device=device)
    return y.squeeze().long().to(device)

class HALE:
    def __init__(self, config, n_classes, d_in, device):
        self.config = config
        self.device = device
        self.n_classes = n_classes
        self.d_in = d_in
        L = config['L']
        J = config['J']
        d_r = config['d_r']
        d_s = config['d_s']
        d_r_total = J * d_r
        self.d_s = d_s
        self.input_proj = torch.randn(d_s, d_in, device=device) * 0.1
        self.reservoirs = [HeterogeneousReservoir(d_in=d_s, d_r=d_r, alpha_list=config['alpha_list'], rho_list=config['rho_list'], device=device) for _ in range(L)]
        self.blocks = [EPBlock(d_s=d_s, d_r_total=d_r_total, clip_alpha=config['clip_alpha'], n_classes=n_classes, device=device) for _ in range(L)]
        self.null_spaces_W = [RecursiveNullSpace(d_total=d_r_total, delta=config['delta'], device=device) for _ in range(L)]
        self.null_spaces_skip = [RecursiveNullSpace(d_total=d_s, delta=config['delta'], device=device) for _ in range(L)]
        self.W_out = torch.zeros(n_classes, d_s, device=device)
        rls_delta = config.get('rls_delta', 1.0)
        self.P_rls = (1.0 / rls_delta) * torch.eye(d_s, device=device)
        self.rls_lambda = config.get('rls_lambda', 1.0)
        self.critic = NeuromdulatoryCtric(kappa=config['kappa'], lam=config['lam'], eps=config['eps'])

    def rls_update(self, h, y):
    # h: (B, d_s) or (d_s,)
    # y: (B,) or scalar
    
    # Normalize shapes
    if h.dim() == 1:
        h = h.unsqueeze(0)
    
    y_tensor = y
    if not isinstance(y_tensor, torch.Tensor):
        y_tensor = torch.tensor([y_tensor], dtype=torch.long,
                                device=self.device)
    y_tensor = y_tensor.view(-1).long()  # always (B,)
    
    # Ensure B matches
    B = h.shape[0]
    if y_tensor.shape[0] != B:
        y_tensor = y_tensor.expand(B)
    
    total_loss = 0.0
    total_correct = 0
    for i in range(B):
        pred_i, loss_i = self._rls_single(h[i], y_tensor[i].item())
        total_correct += pred_i
        total_loss += loss_i
    
    return total_correct / B, total_loss / B

    def _rls_single(self, h_i, y_i):
        with torch.no_grad():
            logits = self.W_out @ h_i
            pred = logits.argmax().item()
            target = torch.zeros(self.n_classes, device=self.device)
            target[y_i.item()] = 1.0
            Ph = self.P_rls @ h_i
            denom = self.rls_lambda + h_i @ Ph
            k = Ph / denom
            self.P_rls = (1.0 / self.rls_lambda) * (self.P_rls - torch.outer(k, Ph))
            error = target - logits
            self.W_out += torch.outer(error, k)
            logits_new = self.W_out @ h_i
            log_probs = torch.log_softmax(logits_new, dim=0)
            loss = -log_probs[y_i.item()].item()
        return (pred == y_i.item()), loss

    def full_train_step(self, x_seq, y):
        # x_seq: (B, T, d_in)
        y = to_scalar_label(y, self.device)
        config = self.config
        L = len(self.blocks)
        d_s = self.d_s
        B = x_seq.shape[0]
        T = x_seq.shape[1]

        r_t_per_block = [None] * L
        s0_per_block = [None] * L
        s_prev2_per_block = [None] * L
        
        self.reset_reservoir_states(B)
        h_outputs = [torch.zeros(B, d_s, device=self.device) for _ in range(L + 2)]

        for t in range(T):
            pixel = x_seq[:, t, :]  # (B, d_in)
            h_outputs[0] = pixel @ self.input_proj.T
            for l in range(L):
                r_t = self.reservoirs[l].step(h_outputs[l])
                r_t_per_block[l] = r_t
                s_prev2_per_block[l] = h_outputs[max(0, l - 1)]
                s0 = self.blocks[l].free_phase_analytical(r_t, s_prev2_per_block[l])
                s0_per_block[l] = s0
                h_outputs[l + 1] = self.blocks[l].get_output(s0)

        h_final = h_outputs[L]
        h_detached = h_final.detach()
        correct, loss_val = self.rls_update(h_detached, y)
        e_t = loss_val
        M_t = self.critic.update(e_t)

        for l in range(L):
            r_t = r_t_per_block[l]
            s0 = s0_per_block[l]
            s_prev2 = s_prev2_per_block[l]
            s_beta = self.blocks[l].nudge_phase(s0, r_t, s_prev2, y, config['beta'], config['gamma'], config['N_nudge'])
            self.blocks[l].update(s_beta, s0, r_t, s_prev2, config['eta'], config['beta'], M_t)
            self.null_spaces_W[l].update(r_t)
            self.null_spaces_skip[l].update(s_prev2)

        return loss_val, correct, M_t

    def task_boundary(self):
        L = len(self.blocks)
        for l in range(L):
            W_task = self.blocks[l].W.clone()
            W_skip_task = self.blocks[l].W_skip.clone()
            self.blocks[l].W = self.null_spaces_W[l].project(self.blocks[l].W, W_task)
            self.blocks[l].W_skip = self.null_spaces_skip[l].project(self.blocks[l].W_skip, W_skip_task)
            self.null_spaces_W[l].reset(self.config['delta'])
            self.null_spaces_skip[l].reset(self.config['delta'])
        rls_delta = self.config.get('rls_delta', 1.0)
        self.P_rls = (1.0 / rls_delta) * torch.eye(self.d_s, device=self.device)
        self.reset_reservoir_states(1)
        self.critic.reset()

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        for x_seq, y in dataloader:
            x_seq = x_seq.to(self.device)
            y = to_scalar_label(y, self.device)
            B = x_seq.shape[0]
            saved_states = [res.get_state_copies() for res in self.reservoirs]
            h_final = self.forward_sequence(x_seq)
            for i, res in enumerate(self.reservoirs):
                res.set_states(saved_states[i])
            with torch.no_grad():
                logits = h_final @ self.W_out.T
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += B
        return correct / total if total > 0 else 0.0

    def forward_sequence(self, x_seq):
        L = len(self.blocks)
        d_s = self.d_s
        B = x_seq.shape[0]
        h_outputs = [torch.zeros(B, d_s, device=self.device) for _ in range(L + 2)]
        self.reset_reservoir_states(B)
        for t in range(x_seq.shape[1]):
            h_outputs[0] = x_seq[:, t, :] @ self.input_proj.T
            for l in range(L):
                r_t = self.reservoirs[l].step(h_outputs[l])
                s0 = self.blocks[l].free_phase_analytical(r_t, h_outputs[max(0, l - 1)])
                h_outputs[l + 1] = self.blocks[l].get_output(s0)
        return h_outputs[L]

    def save_checkpoint(self, path, task_k, metrics):
        torch.save({'task': task_k, 'metrics': metrics, 'W_out': self.W_out.cpu(), 'P_rls': self.P_rls.cpu(), 'blocks_W': [b.W.cpu() for b in self.blocks], 'blocks_W_skip': [b.W_skip.cpu() for b in self.blocks], 'null_W': [n.P.cpu() for n in self.null_spaces_W], 'null_skip': [n.P.cpu() for n in self.null_spaces_skip]}, path)
        print(f'Checkpoint saved: task {task_k} -> {path}')

    def reset_reservoir_states(self, B=1):
        for r in self.reservoirs:
            r.reset(B)
