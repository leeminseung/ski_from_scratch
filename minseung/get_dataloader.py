from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class SKIDataset(nn.Module):
    def __init__(self, dataset):
        super(SKIDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx, :-1,:], self.dataset[idx, -1, :-1]

    def __len__(self):
        return self.dataset.shape[0]

def get_dataloader(dataset, device, args):
    # pin_memory = False if device=='cpu' else True
    pin_memory = False
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = dataset.astype(np.float32)

    train_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=42)
    train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

    d4rl_train_dataset = SKIDataset(train_data)
    d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_val_dataset = SKIDataset(validation_data)
    d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_test_dataset = SKIDataset(test_data)
    d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)
    return d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader
