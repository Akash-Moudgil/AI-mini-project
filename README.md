# Classifying Apple Varieties

## Classifying Apple Varieties with Deep Learning: A Step-by-Step Guide

### Introduction<br>
Fruit classification has become essential in agriculture and retail, helping automate the sorting and grading of produce for quality control. In this project, we’ll build a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify five types of apples: **Fuji, Golden Delicious, Granny Smith, Lady, and Red Delicious.** This guide will cover dataset preparation, data augmentation, CNN architecture, training, evaluation, and testing. By the end, you’ll have a robust model capable of distinguishing apple varieties from images with high accuracy.

#### 1. Dataset Preparation:-<br>
To prepare our data, we organized it into separate folders for training and validation, each containing images for the five apple types. This structured format allows TensorFlow to efficiently load and preprocess images, maintaining consistency in data input.
**Training directory:** *C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Train*
**Validation directory:** *C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Validation*

Each folder within these directories corresponds to a unique apple variety, making it easier for the CNN model to map image features to specific labels during training.
