# Tactile Intelligence

## Overview
This project explores tactile perception for robotics using a novel dual classification framework.  
The goal is to predict two distinct object properties — **surface texture** and **material rigidity** — 
from multichannel tactile signals captured during robotic grasp interactions.  

To benchmark performance, both classical machine learning methods (Support Vector Machines) 
and deep learning architectures (CNN, CAE, LSTM) are implemented and evaluated under 
In-Distribution and Out-of-Distribution conditions.  

The project also introduces a curated dataset of tactile signals collected from a custom 
bio-inspired magnetic sensor, providing a challenging benchmark for multi-task tactile learning. 

## Installation
Install dependecies, ensure that pytorch is properly installed beforehand
```bash
git clone <https://github.com/IvoKosa/tactile_intelligence>
cd <tactile_intelligence>
pip install -r requirements.txt
```

## Training and Testing
manager.py      -- Runs the training and testing for the CNN, CAE and LSTM models
                -- All function parameters can be found within the file comments
                -- Simply change any desired parameters in the file and run

signal_SVM.py   -- Runs the SVM models
                -- Parameters specified in file
                -- this file can be run as is

## Previous experiments
All data used in the report along with model weights can be found in the FINAL_... folders

## Dataset
The full dataset is stored under data_final which itself is split into two folders: multigrasp_train (Multi-grasp collection 1 & single-grasp) and multigrasp_test (Multi-grasp collection 2) for easier use within the training and testing programs. The single-grasp collection is indicated with the folders ending in _200 and the multi-grasp collections have been marked with the folders ending in _multigrasp. The folders are arranged in this manner due to the unique train-test split of the data, allowing for Out-of-distribution testing. 

Each individual grasp is stored in a hierachical folder system which encodes all the necessary information regarding texture, material and additional info such as position of the gripper for the multi-grasp data. Within these folders each grasp is represented by 3 seperate csv files sensor0_data_trial_code.csv, sensor1_data_trial_code.csv and gripper_positions_trial_n.csv (gripper positions are not used in this project). The trial codes are unique identifiers for the individual grasps and are identical for sensor 0 and sensor 1, whereas the gripper positions are identified by integers and are matched to the sensors in numerical order. These files are accumulated by the signal_dataset.py class (inherits from torch.utils.data.Dataset) which handels all class identification and file matching to gather all information needed for a single grasp (primarily handled by the function utils.collect_file_info()). Details of the parameters used within the SignalDataset class are found in the manager.py script, which handles everything used for testing and training. This allows for simple integration with PyTorch within custom models and training/ testing scripts. 

The data gathered from the grasps is stored in the sensor0 and sensor1 csv files with the following headers:

```timestamp,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x1_calib,y1_calib,z1_calib,x2_calib,y2_calib,z2_calib,x3_calib,y3_calib,z3_calib,x4_calib,y4_calib,z4_calib```

Each row represents the readings of each axis of the sensor for each Taxel grouped into x_n, y_n, z_n, where n shows the Taxel number for 4 total Taxels per sensor. Both the raw values and calibrated values (ending in _calib) are collected, where the sensor readings are re-calibrated to start at 0 before each grasp. Only the calibrated values were used for the results gained in this project. The gripper positions show the gripper appetrure (angle) at a selection of timestamps and were not used ito gather any of the results shown.

## Videos:

Videos of the data collection process can be viewed here: https://drive.google.com/drive/folders/1O33RIPMjevWZI-CN7F8SJx75xm_70W13?usp=sharing