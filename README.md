# Evaluating Conformal Prediction Under Distribution Shift and Long-tailed Data

This repository contains code for evaluating conformal prediction results on OOD ImageNet datasets. 

**main.py** runs conformal prediction + evaluation on IN1k and IN variants. The conformal and evaluation methods are implemented in **conforma_prediction.py** and **evaluation.py**, respectively.

To make this process more efficient, inference results (softmax values) from various models are pre-computed and saved. The code for this is implemented in **save_results/**.  
