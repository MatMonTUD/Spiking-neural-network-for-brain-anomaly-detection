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

def save_model(net, save_flag, model_name):
    if save_flag == 1:
        os.makedirs("../Models", exist_ok=True)
        if not model_name.endswith(".pth"):
            model_name += ".pth"
        filepath = os.path.join("../Models", model_name)
        # Save the entire model, including its architecture and weights
        torch.save(net, filepath)
        print(f"Model saved to {filepath}")

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

#It takes 200+ epochs but it gets to a very good accuracy
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