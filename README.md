# Explainable Computer Vision — COVID-19 Chest X-Ray Classification

## Overview
The project trains two image classification models ResNet18 & ViT-B/16
on the COVID-19 Dataset to classify chest X-rays into three 
categories: COVID, Normal, and Viral Pneumonia. 
Explainability methods used - GradCAM, GradCAM++, ScoreCAM, Attention Rollout, Attribution Maps 
They have been applied and evaluated using Entropy, AOPC, and Insertion/Deletion metrics.

## Results
| Model     | Accuracy | Precision | Recall | F1   |
|-----------|----------|-----------|--------|------|
| ResNet18  | 0.99     | 0.99      | 0.99   | 0.99 |
| ViT-B/16  | 0.98     | 0.98      | 0.98   | 0.98 |

## How to Run
1. Open ExCV.ipynb in Google Colab
2. Enable GPU (by setting runtime type to T4 GPU)
3. Mount your Google Drive (by editing cell 1) and upload the dataset  
4. Run all cells in order
   
## Dataset
COVID-19 Chest X-Ray dataset provided by AIMS-DTU

## Outputs
Model weights, saliency maps, and visualizations can be viewed at :
[Google Drive]
(https://drive.google.com/drive/folders/1mCvkmk79Qir_eb2P1wMRMQddoRKUaZNL?usp=sharing)

## Requirements, Imports
torch, torchvision, grad-cam, scipy, sklearn, matplotlib, opencv-python, numpy, Pillow, pandas
