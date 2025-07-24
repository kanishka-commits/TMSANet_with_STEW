# ---- Data Settings ---- #
dataset = 'stew'
data_path = './STEW_Dataset'
ratings_file = './STEW_Dataset/ratings.txt'
# train_files = ['training.mat']
# test_files = ['evaluation.mat']

batch_size = 4  # increase to reduce variance
num_segs = 8     # Number of segments for augmentation

# ---- Model Settings ---- #
pool_size = 50
pool_stride = 15
num_heads = 4
fc_ratio = 2
depth = 1

# ---- Training Settings ---- #
epochs = 10
learning_rate = 5e-5  # lower LR to stabilize
weight_decay = 1e-3   # increase regularization

#initially it was learning_rate = 1e-4; weight_decay = 1e-4


# ---- Dropout Settings ---- #
attn_drop = 0.6  # increase dropout to prevent overfitting
fc_drop = 0.6

# ---- Output Settings ---- #
output = 'output'
model_name = "TMSANet_STEW"