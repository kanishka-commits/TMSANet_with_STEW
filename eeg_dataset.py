import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class STEWDataset(Dataset):
    """
    EEG dataset loader for STEW workload dataset.
    Loads EEG signals from text files, normalizes channels,
    attaches task/rest labels from ratings file.
    """

    def __init__(self, data_dir, ratings_file, mode="task"):
        """
        Args:
            data_dir (str): Path to folder containing EEG text files.
            ratings_file (str): Path to ratings.txt.
            mode (str): 'task' or 'rest' to determine labeling.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.subject_labels = {}

        # Read subject ratings
        with open(ratings_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    subno, rest_rating, task_rating = map(int, parts)
                    rating = rest_rating if mode == "rest" else task_rating
                    if rating <= 3:
                        label = 0
                    elif rating <= 6:
                        label = 1
                    else:
                        label = 2
                    self.subject_labels[subno] = label
                except ValueError:
                    continue  # skip bad lines

        # Build list of (file_path, label)
        self.file_list = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".txt") and fname.startswith("sub"):
                try:
                    subno = int(fname.split("_")[0].replace("sub", ""))
                    if subno in self.subject_labels:
                        if (mode == "rest" and "_lo" in fname) or (mode == "task" and "_hi" in fname):
                            fpath = os.path.join(data_dir, fname)
                            label = self.subject_labels[subno]
                            self.file_list.append((fpath, label))
                except Exception:
                    continue

        # Shuffle to avoid order bias
        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        data = np.loadtxt(file_path).T  # shape (14, 19200)
        
        # Normalize each channel
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
        
        # Convert to (1, C, T) for torch
        data = data[np.newaxis, :, :]  
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return data, label
