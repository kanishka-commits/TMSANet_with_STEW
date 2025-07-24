import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import config
from eeg_dataset import STEWDataset
from network.TMSANet import TMSANet
import train  # assumes your train.py has train_evaluation()

# For reproducibility
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your STEW dataset
dataset = STEWDataset(config.data_path, config.ratings_file, mode="task")

# Split into train and test sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize your TMSA-Net model
model = TMSANet(
    in_planes=14,       # EEG has 14 channels
    radix=1,
    time_points=19200,  # 19200 time points (2.5 min @128Hz)
    num_classes=3,
    embed_dim=19,
    pool_size=config.pool_size,
    pool_stride=config.pool_stride,
    num_heads=config.num_heads,
    fc_ratio=config.fc_ratio,
    depth=config.depth,
    attn_drop=0.5,
    fc_drop=0.5
)
model = model.to(device)

# Loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Directory to save outputs
save_path = "./output"

# Train and evaluate
best_acc, best_kappa = train.train_evaluation(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    save_path,
    epochs=config.epochs
)

print(f"\n✅ Finished training. Best Accuracy: {best_acc:.4f}, Best Kappa: {best_kappa:.4f}")

