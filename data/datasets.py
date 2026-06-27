import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class SequentialDataset(Dataset):
    def __init__(self, dataset_name, task_id, train, n_classes_per_task, d_in, T, max_samples=None):
        self.d_in = d_in
        self.T = T
        self.n_classes_per_task = n_classes_per_task
        self.task_classes = list(range(task_id * n_classes_per_task, task_id * n_classes_per_task + n_classes_per_task))
        transform = transforms.Compose([transforms.ToTensor()])
        if dataset_name == 'MNIST':
            base = torchvision.datasets.MNIST(root='./data_cache', train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR10':
            base = torchvision.datasets.CIFAR10(root='./data_cache', train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR100':
            base = torchvision.datasets.CIFAR100(root='./data_cache', train=train, download=True, transform=transform)
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
        indices = [i for i, (_, label) in enumerate(base) if label in self.task_classes]
        if max_samples:
            indices = indices[:max_samples]
        self.data = [base[i][0] for i in indices]
        self.labels = [self.task_classes.index(base[i][1]) for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = (img - 0.5) / 0.5
        if img.shape[0] == 1:
            x_seq = img.squeeze(0).reshape(self.T, self.d_in)
        else:
            x_seq = img.permute(1, 2, 0).reshape(self.T, self.d_in)
        return x_seq.float(), torch.tensor(self.labels[idx], dtype=torch.long)

def build_loaders(dataset_key, dcfg):
    train_loaders, test_loaders = [], []
    for k in range(dcfg['n_tasks']):
        tr = SequentialDataset(dcfg['dataset'], k, True, dcfg['n_classes_per_task'], dcfg['d_in'], dcfg['T'], dcfg.get('train_samples'))
        te = SequentialDataset(dcfg['dataset'], k, False, dcfg['n_classes_per_task'], dcfg['d_in'], dcfg['T'], dcfg.get('test_samples'))
        train_loaders.append(DataLoader(
            tr, batch_size=64, shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        ))
        test_loaders.append(DataLoader(
            te, batch_size=64,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        ))
    return train_loaders, test_loaders
