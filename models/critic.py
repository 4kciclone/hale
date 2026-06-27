import numpy as np

class NeuromdulatoryCtric:
    def __init__(self, kappa, lam, eps):
        self.kappa = kappa
        self.lam = lam
        self.eps = eps
        self.e_bar = 0.0
        self.sigma2 = 0.1
        self.history = []

    def update(self, e_t):
        self.e_bar = (1 - self.lam) * self.e_bar + self.lam * e_t
        self.sigma2 = (1 - self.lam) * self.sigma2 + self.lam * (e_t - self.e_bar) ** 2
        z = self.kappa * (e_t - self.e_bar) / (np.sqrt(self.sigma2) + self.eps)
        M_t = 1.0 / (1.0 + np.exp(-z))
        self.history.append(M_t)
        return float(M_t)

    def reset(self):
        self.e_bar = 0.0
        self.sigma2 = 0.1
        self.history = []
