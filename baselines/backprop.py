import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scalar_label(y, device):
    if not isinstance(y, torch.Tensor):
        return torch.tensor(y, dtype=torch.long, device=device)
    y = y.squeeze()
    if y.dim() == 0:
        return y.long().to(device)
    return y.view(-1)[0].long().to(device)

class StandardBackprop:
    def __init__(self, d_in, T, n_classes, device):
        self.device = device
        self.d_in = d_in
        self.T = T
        self.d_in_flat = d_in * T
        self.net = nn.Sequential(
            nn.Linear(self.d_in_flat, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        ).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, x_seq, y):
        x_flat = x_seq.reshape(-1).unsqueeze(0).to(self.device)
        label = to_scalar_label(y, self.device)
        self.opt.zero_grad()
        output = self.net(x_flat)
        loss = F.cross_entropy(output, label.unsqueeze(0))
        loss.backward()
        self.opt.step()
        pred = output.argmax(dim=1).item()
        return (pred == label.item()), loss.item()

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for x_seq, y in dataloader:
                x_flat = x_seq.reshape(-1).unsqueeze(0).to(self.device)
                label = to_scalar_label(y, self.device)
                output = self.net(x_flat)
                pred = output.argmax(dim=1).item()
                if pred == label.item():
                    correct += 1
                total += 1
        self.net.train()
        return correct / total if total > 0 else 0.0
