import os
import numpy as np
from torch.utils.data import Dataset
import torch

class STEWDataset(Dataset):
    def __init__(self, data_dir, window_size=1000, stride=500, transform=None):
        #window_size: length (in time points) of each data segment (default = 1000)
        #stride: how far to slide the window forward each time (default = 500)
        #transform: optional function for data augmentation or preprocessing
        self.samples = []
        self.labels = []
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        for fname in os.listdir(data_dir):
            if not fname.endswith(".txt"):
                continue
            label = 1 if fname.endswith("_hi.txt") else 0
            path = os.path.join(data_dir, fname)
            data = np.loadtxt(path)  # shape: (time, channels)
            #Loads the EEG file into a NumPy array

            # Windowing
            for start in range(0, data.shape[0] - window_size + 1, stride):
                segment = data[start:start + window_size]
                self.samples.append(segment)
                self.labels.append(label)

        self.samples = np.stack(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].astype(np.float32).T  # shape: (channels, time)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x), torch.tensor(y)
