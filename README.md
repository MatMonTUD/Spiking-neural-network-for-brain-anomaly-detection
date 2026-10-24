Here, we will explain how to use the implementation code. But For now, I will just paste the instructions found in "Project11_snn_brain.pdf"

# Semester Project 11: Spiking Neural Network for Brain Anomaly Detection

## Topic

Detecting brain anomalies using spiking neural networks.

## Task

Developing spiking neural networks to classify Electroencephalogram (EEG) segments.

## Supervisor

Dr. Hongliu Yang, [hongliu.yang@tu-dresden.de](mailto:hongliu.yang@tu-dresden.de)

## Dataset

Intracranial EEG was recorded from dogs with naturally occurring epilepsy. Data are organized in folders containing training and testing data for each canine subject. The training data consist of 1-second EEG clips labeled "Ictal" (seizure segments) or "Interictal" (non-seizure segments). For this study, 80% of the provided training data will be used for model training and the remaining 20% for testing.

Link to dataset: https://www.kaggle.com/competitions/seizure-detection/data

## snnTorch

snnTorch is a PyTorch compatible platform for spiking neural networks.

Link to snnTorch: https://snntorch.readthedocs.io/en/latest/

## Task description

1. Read the literature on spiking neural networks [1].
2. Download the dataset and split the data into training and testing sets for each subject. Visualize selected EEG segments to develop an understanding of the difference between brain states.
3. Learn to use PyTorch's Dataset and Dataloader classes to construct your custom EEG dataset.
4. Install snnTorch and experiment with developing spiking neural networks using provided examples.
5. Construct a 3-layer CNN (conv5x8, conv5x16, linear) with LIF neurons. Train it on the MNIST dataset and evaluate its performance. Experiment with different spike encoding methods and loss functions.
6. Construct a 6-layer CNN (conv5x8, conv5x16, 3xconv5x16, linear) with LIF neurons. Train it on the EEG dataset for dog 1 using rate coding and evaluate the performance. Analyze how performance varies with different random initializations.
7. Compare the performance of the spiking neural network with that of the corresponding ANN.
8. Prepare your results in the form of presentation slides for the colloquium at the end of the semester.
9. Comment your implementation code so that it is understandable even without a detailed report. Add a Readme file that explains how to use your implementation code. Submit the commented implementation code as one ZIP file to your supervisor before the examination.

## Literature

[1] Eshraghian, Jason K., et al. "Training spiking neural networks using lessons from deep learning." Proceedings of the IEEE 111.9 (2023): 1016-1054. https://doi.org/10.48550/arXiv.2109.12894
