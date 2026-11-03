import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil

SourceData = f"../dog_1"
TestData = f"../Test_data"
TrainData = f"../Train_data"

os.makedirs(TestData, exist_ok=True)
os.makedirs(TrainData, exist_ok=True)

files = os.listdir(SourceData)
random.shuffle(files)

split = int(len(files) * 0.8)
TrainCount = 0
TestCount = 0
for file in files[:split]:
    shutil.copy(os.path.join(SourceData, file), os.path.join(TrainData, file))
    TrainCount +=1
for file in files[split:]:
    shutil.copy(os.path.join(SourceData, file), os.path.join(TestData, file))
    TestCount +=1
print("Test files count: ", TestCount)
print("Train files count: ", TrainCount)
