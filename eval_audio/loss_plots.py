import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
loss_file = "/home/ixzhu/Qwen-Audio/checkpoints_MLP/checkpoint_epoch_2/losses.txt"      # path to your loss file
loss_file_2 = "/home/ixzhu/Qwen-Audio/checkpoints_CNN/checkpoint_epoch_2/losses.txt"
smooth_window = 1000            # moving average window size
num_epochs = 2
out_file = "/home/ixzhu/Qwen-Audio/eval_audio/results/loss_plot.png"    # output figure path

# -----------------------
# Load losses
# -----------------------
losses = np.loadtxt(loss_file)  # assumes 1 loss per line
losses_2 = np.loadtxt(loss_file_2)
N = len(losses)

# Split into epochs
assert N % num_epochs == 0, "Loss file length must be divisible by number of epochs."
steps_per_epoch = N // num_epochs

# Create x-axis in epoch units (epoch 1 spans 0→1, epoch 2 spans 1→2)
epoch_axis = np.linspace(0, num_epochs, N)

# -----------------------
# Smoothing (moving average)
# -----------------------
def smooth(arr, window):
    if window < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

smooth_losses = smooth(losses, smooth_window)
smooth_losses_2 = smooth(losses_2, smooth_window)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10, 5))
plt.plot(epoch_axis[smooth_window:-smooth_window], smooth_losses[smooth_window:-smooth_window], linewidth=2, label='MLP')
plt.plot(epoch_axis[smooth_window:-smooth_window], smooth_losses_2[smooth_window:-smooth_window], linewidth=2, label='CNN')

# Epoch boundaries
for e in range(1, num_epochs):
    plt.axvline(e, color='gray', linestyle='--', alpha=0.5)

plt.title("Training Loss Over Epochs (Smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save instead of show
plt.savefig(out_file, dpi=300)
plt.close()

print(f"Saved plot to {out_file}")
