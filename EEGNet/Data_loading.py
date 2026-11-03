import os
import random
import shutil

#__This script splits the dataset into train and test__#

# Set the seed
random.seed(42)
#To use the data for training and visualisation:
# extract dog_1.zip into the folder containing this file
# run Data_loading
#The files from dog_1 will be randomly copied into either
# Test_data (20%) or Train_data (80%)

###If you change the folders' name, make sure to add them to .gitignore ###
###                to avoid pushing redundant files on github           ###
SourceData = f"dog_1"
TestData = f"Test_data"
TrainData = f"Train_data"
#Create the folders if they do not exist
os.makedirs(TestData, exist_ok=True)
os.makedirs(TrainData, exist_ok=True)

files = os.listdir(SourceData)
random.shuffle(files) #make sure that the files are copied randomly to Test or Train

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
