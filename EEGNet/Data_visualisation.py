import os
import random
import scipy.io
import matplotlib.pyplot as plt

TrainData = "../Train_data"

def print_class_distribution(data_dir):
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    ictal_count = sum(1 for f in files if "ictal" in f and "interictal" not in f)
    interictal_count = sum(1 for f in files if "interictal" in f)
    print(f"Dataset Path: {data_dir}")
    print(f"  - Interictal (0): {interictal_count}")
    print(f"  - Ictal (1): {ictal_count}")
    print(f"  - Total Files: {len(files)}")

print_class_distribution(TrainData)
print("-" * 40)

files = [f for f in os.listdir(TrainData) if f.endswith('.mat')]
random_file = random.choice(files)
file_path = os.path.join(TrainData, random_file)

if "ictal" in random_file and "interictal" not in random_file:
    class_name = "Ictal (Seizure)"
    label = 1
else:
    class_name = "Interictal (Normal)"
    label = 0

print(f"File Name: {random_file}")
print(f"Class: {class_name} ({label})")

eeg_matrix = scipy.io.loadmat(file_path)['data']

num_channels = eeg_matrix.shape[0]
fig, axes = plt.subplots(num_channels, 1, figsize=(12, 10), sharex=True)

for i in range(num_channels):
    axes[i].plot(eeg_matrix[i], linewidth=0.5)
    axes[i].set_ylabel(f"Ch {i+1}", fontsize=8)
    axes[i].tick_params(labelsize=6)

plt.suptitle(f"File: {random_file} | Class: {class_name}", fontsize=12)
plt.xlabel("Time Samples", fontsize=10)
plt.tight_layout()
plt.show()