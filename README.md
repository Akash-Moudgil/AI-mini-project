# Classifying Apple Varieties

## Classifying Apple Varieties with Deep Learning: A Step-by-Step Guide

### Introduction<br>
Fruit classification has become essential in agriculture and retail, helping automate the sorting and grading of produce for quality control. In this project, we’ll build a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify five types of apples: **Fuji, Golden Delicious, Granny Smith, Lady, and Red Delicious.** This guide will cover dataset preparation, data augmentation, CNN architecture, training, evaluation, and testing. By the end, you’ll have a robust model capable of distinguishing apple varieties from images with high accuracy.

#### 1. Dataset Preparation:-<br>
To prepare our data, we organized it into separate folders for training and validation, each containing images for the five apple types. This structured format allows TensorFlow to efficiently load and preprocess images, maintaining consistency in data input.<br>
**Training directory:** *C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Train*<br>
**Validation directory:** *C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Validation*<br>
Each folder within these directories corresponds to a unique apple variety, making it easier for the CNN model to map image features to specific labels during training.

#### Data Augmentation:-<br>
To enhance our model’s ability to generalize, we apply data augmentation to artificially expand our training dataset. Data augmentation generates varied transformations of each image, reducing overfitting and helping the model learn more robust features. Key transformations include:

Rotation: Randomly rotates images up to 40 degrees to simulate different viewing angles.
Width and Height Shifts: Shifts images along the x and y axes by 20%, adding positional variance.
Shear Transformation: Introduces a shearing effect, altering the image’s shape slightly.
Zooming: Randomly zooms in by 20% to simulate closer viewpoints.
Horizontal Flip: Flips images horizontally to simulate images from both left and right views.
