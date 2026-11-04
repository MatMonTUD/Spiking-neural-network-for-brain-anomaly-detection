import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from snntorch import functional as SF
import Varius as V
from Spiking_EEGNet_models import num_steps, forward_pass, batch_accuracy
import Spiking_EEGNet_models as spm

# Set the seed
V.set_seed(42)

# Setup Device & Hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 64
num_epochs = 250
lr = 1e-3

### Define Network Architecture ###
factory = spm.get_quant_EEGNet_nano
net = factory().to(device)
net_name = factory.__name__.removeprefix("get_")
#net_name = "Model_name" #Use this instead of the line above for a personalised name

# Dataset (raw amplitudes, no normalization)
#Maybe normalizing the inputs can be helpful, idk
class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load .mat file and extract 'data' matrix
        mat_data = scipy.io.loadmat(file_path)
        eeg_matrix = mat_data['data'].astype('float32')

        # Determine label from filename
        label = 1 if "ictal" in file_name and "interictal" not in file_name else 0

        # [16, 400] -> [1, 16, 400]
        data_tensor = torch.tensor(eeg_matrix, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor

#Might want to create a val set to avoid any bias
#train_dataset = EEGDataset("../Train_data")
#test_dataset = EEGDataset("../Test_data")
train_dataset = EEGDataset("Train_data")
test_dataset = EEGDataset("Test_data")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Training Configuration
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

# Weight the classes because one is much more represented
#Not sure if it's really needed, also the ratio might change
class_weights = torch.tensor([0.70, 1.74], dtype=torch.float32).to(device)
loss_fn = SF.ce_rate_loss(weight=class_weights)

# Training Loop
#Print model's parameters
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model Name:", net_name)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

print("Starting Training...")
counter = 0
test_acc = None
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    test_acc = batch_accuracy(test_loader, net, num_steps, device)
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device)

        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)
        loss_val = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        """
        if counter % 15 == 0:
            test_acc = batch_accuracy(test_loader, net, num_steps, device)
        """
        info = {"loss": f"{loss_val.item():.4f}"}
        if test_acc is not None:
            info["test_acc"] = f"{test_acc * 100:.2f}%"
        pbar.set_postfix(info)

        counter += 1
save_flag = 0 #set to 1 if you want to save the model
#at the moment it gives an error because it doesn't recognize surrogate.atan or something like that
spm.save_model(net,  save_flag, model_name=net_name) # DON'T FORGET TO CHANGE THE NAME!!
print(f"Final test accuracy: {batch_accuracy(test_loader, net, num_steps, device) * 100:.2f}%")