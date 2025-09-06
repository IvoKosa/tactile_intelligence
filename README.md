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
Outline any dependencies or setup instructions. For example:  
```bash
git clone <https://github.com/IvoKosa/tactile_intelligence/tree/main>
cd <tactile_intelligence>
pip install -r requirements.txt
```

## Training and Testing
manager.py      -- Run the training and testing for the CNN, CAE and LSTM models
                -- All function parameters can be found within the file comments

signal_SVM.py   -- Runs the SVM models
