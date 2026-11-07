import torch
import torch.nn as nn
import brevitas.nn as qnn
import os

#__This script contains some functions used to train the non-spiking models, __#
#__        together with the architecture description of said models         __#

def forward_pass(net, data):
    return net(data)

def batch_accuracy(loader, net, device):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            out = forward_pass(net, data)
            acc += (out.argmax(dim=1) == targets).float().mean().item() * out.size(0)
            total += out.size(0)
    return acc / total

#This code doesn't work well with spiking and quantised models
def save_model(net, save_flag, model_name):
    if save_flag == 1:
        os.makedirs("../Models", exist_ok=True)
        if not model_name.endswith(".pth"):
            model_name += ".pth"
        filepath = os.path.join("../Models", model_name)
        # Save the entire model, including its architecture and weights
        torch.save(net, filepath)
        print(f"Model saved to {filepath}")

@torch.no_grad()
def evaluate_confusion(loader, net, device, num_classes=2):
    net.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)  # rows: true, cols: predicted
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        preds = forward_pass(net, data).argmax(dim=1)
        for t, p in zip(targets.cpu(), preds.cpu()):
            cm[t, p] += 1
    return cm

def report_results(net, loader, device, net_name, seed, num_epochs, total_params):
    cm = evaluate_confusion(loader, net, device)
    final_acc = cm.diag().sum().item() / cm.sum().item()

    # Row-normalized: each row sums to 100%
    row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)  # avoid division by zero
    cm_pct = cm.float() / row_sums * 100

    print("\n" + "=" * 40)
    print(f"Model:            {net_name}")
    print(f"Seed:             {seed}")
    print(f"Epochs:           {num_epochs}")
    print(f"Total parameters: {total_params:,}")
    print(f"Test accuracy:    {final_acc * 100:.2f}%")
    print("Confusion matrix, % of each actual class (rows = actual, cols = predicted):")
    print("                  pred interictal   pred ictal")
    print(f"  actual interictal   {cm_pct[0, 0].item():10.2f}%  {cm_pct[0, 1].item():10.2f}%")
    print(f"  actual ictal        {cm_pct[1, 0].item():10.2f}%  {cm_pct[1, 1].item():10.2f}%")
    print("=" * 52)

    return cm, cm_pct, final_acc

#This is the architecture inspired by EEGNet.
# It's actually a bit different, to adapt it to the data used here
def get_EEGNet():
    return nn.Sequential(
        # Block 1
        nn.Conv2d(1, 8, (1, 200), padding='same', bias=True),
        nn.Conv2d(8, 16, (16, 1), groups=8, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(0.5),
        # Block 2
        nn.Conv2d(16, 16, (1, 16), padding='same', groups=16, bias=True),
        nn.Conv2d(16, 16, 1, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(0.5),
        # Classifier
        nn.Flatten(),
        nn.Linear(16 * (400 // 32), 2),
    )

#It's just like EEGNet above, but the number of channels used in all the convolutions
# is much smaller so that its size is reduced by about 5 times
def get_EEGNet_nano():
    return nn.Sequential(
        # Block 1
        nn.Conv2d(1, 2, (1, 200), padding='same', bias=True),
        nn.Conv2d(2, 2, (16, 1), groups=2, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(0.5),
        # Block 2
        nn.Conv2d(2, 2, (1, 16), padding='same', groups=2, bias=True),
        nn.Conv2d(2, 2, 1, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(0.5),
        # Classifier
        nn.Flatten(),
        nn.Linear(2 * (400 // 32), 2),
    )

#Here are the models seen above, but quantised.
#So that we can check how robust are SNN to quantisation
#Even when 2bits-quantised the results are quite good. It takes 100+ epochs to get there tho
def get_quant_EEGNet():
    bits = 2
    return nn.Sequential(
        # Block 1
        qnn.QuantConv2d(1, 8, (1, 200), padding='same', weight_bit_width=bits, bias=True),
        qnn.QuantConv2d(8, 16, (16, 1), groups=8, weight_bit_width=bits, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(0.5),
        # Block 2
        qnn.QuantConv2d(16, 16, (1, 16), padding='same', groups=16, weight_bit_width=bits, bias=True),
        qnn.QuantConv2d(16, 16, 1, weight_bit_width=bits, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(0.5),
        # Classifier
        nn.Flatten(),
        qnn.QuantLinear(16 * (400 // 32), 2, weight_bit_width=bits, bias=True),
    )

#It takes 200+ epochs, but it reaches a very good accuracy
def get_quant_EEGNet_nano():
    bits = 2
    return nn.Sequential(
        # Block 1
        qnn.QuantConv2d(1, 2, (1, 200), padding='same', weight_bit_width=bits, bias=True),
        qnn.QuantConv2d(2, 2, (16, 1), groups=2, weight_bit_width=bits, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(0.5),
        # Block 2
        qnn.QuantConv2d(2, 2, (1, 16), padding='same', groups=2, weight_bit_width=bits, bias=True),
        qnn.QuantConv2d(2, 2, 1, weight_bit_width=bits, bias=True),
        nn.ReLU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(0.5),
        # Classifier
        nn.Flatten(),
        qnn.QuantLinear(2 * (400 // 32), 2, weight_bit_width=bits, bias=True),
    )