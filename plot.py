import re
import matplotlib.pyplot as plt
import numpy as np

log_path = "log.txt"   # Replace with your log file name

# These two regex patterns correspond to:
# 1. The epoch line (containing box_loss / cls_loss / dfl_loss)
# 2. The "all" line (containing P / R / mAP50 / mAP50-95)
pattern_epoch = re.compile(
    r'^\s*(\d+)/\d+\s+[0-9.]+G\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
)
pattern_all = re.compile(
    r'^\s*all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
)

epochs = []
box_losses = []
cls_losses = []
dfl_losses = []
P_list = []
R_list = []
mAP50_list = []
mAP5095_list = []

with open(log_path, encoding="utf-8") as f:
    for line in f:
        # Extract epoch + losses
        m = pattern_epoch.search(line)
        if m:
            epoch = int(m.group(1))
            box_loss = float(m.group(2))
            cls_loss = float(m.group(3))
            dfl_loss = float(m.group(4))

            epochs.append(epoch)
            box_losses.append(box_loss)
            cls_losses.append(cls_loss)
            dfl_losses.append(dfl_loss)
            continue

        # Extract evaluation metrics (the "all" line)
        # Ensure each epoch corresponds to only one set of metrics
        m = pattern_all.search(line)
        if m and len(P_list) < len(epochs):
            P_list.append(float(m.group(1)))
            R_list.append(float(m.group(2)))
            mAP50_list.append(float(m.group(3)))
            mAP5095_list.append(float(m.group(4)))

print("Total epochs captured:", len(epochs))
print("Loss lengths:", len(box_losses), len(cls_losses), len(dfl_losses))
print("Metric lengths:", len(P_list), len(R_list), len(mAP50_list), len(mAP5095_list))

# Quick check: all lengths should be equal
assert len(epochs) == len(box_losses) == len(cls_losses) == len(dfl_losses) == len(P_list)

plt.figure()
plt.plot(epochs, box_losses, label="box_loss")
plt.plot(epochs, cls_losses, label="cls_loss")
plt.plot(epochs, dfl_losses, label="dfl_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(epochs, P_list, label="Precision (P)")
plt.plot(epochs, R_list, label="Recall (R)")
plt.plot(epochs, mAP50_list, label="mAP50")
plt.plot(epochs, mAP5095_list, label="mAP50-95")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mAP50_arr = np.array(mAP50_list)
best_idx = int(mAP50_arr.argmax())

print(f"Best mAP50 = {mAP50_list[best_idx]:.3f} at epoch {epochs[best_idx]}")
print(f"Precision = {P_list[best_idx]:.3f}, Recall = {R_list[best_idx]:.3f}, mAP50-95 = {mAP5095_list[best_idx]:.3f}")
