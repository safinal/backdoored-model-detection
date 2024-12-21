import torch

seed = 0
img_resize = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = 100
batch_size = 30
initial_lr = 0.1
final_lr = 0.001
warmup_steps = 10