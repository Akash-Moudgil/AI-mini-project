# Classifying Apple Varieties

## Classifying Apple Varieties with Deep Learning: A Step-by-Step Guide

### Introduction<br>
Fruit classification has become essential in agriculture and retail, helping automate the sorting and grading of produce for quality control. In this project, we’ll build a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify five types of apples: **Fuji, Golden Delicious, Granny Smith, Lady, and Red Delicious.** This guide will cover dataset preparation, data augmentation, CNN architecture, training, evaluation, and testing. By the end, you’ll have a robust model capable of distinguishing apple varieties from images with high accuracy.
***
### 1. Dataset Preparation:-<br>
To prepare our data, we organized it into separate folders for training and validation, each containing images for the five apple types. This structured format allows TensorFlow to efficiently load and preprocess images, maintaining consistency in data input.<br>
**Training directory:** ```pythonC:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Train```<br>
**Validation directory:** ```pythonC:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Validation```<br>
Each folder within these directories corresponds to a unique apple variety, making it easier for the CNN model to map image features to specific labels during training.
***
### 2. Data Augmentation:-<br>
To enhance our model’s ability to generalize, we apply data augmentation to artificially expand our training dataset. Data augmentation generates varied transformations of each image, reducing overfitting and helping the model learn more robust features. Key transformations include:

**Rotation:** Randomly rotates images up to 40 degrees to simulate different viewing angles.<br>
**Width and Height Shifts:** Shifts images along the x and y axes by 20%, adding positional variance.<br>
**Shear Transformation:** Introduces a shearing effect, altering the image’s shape slightly.<br>
**Zooming:** Randomly zooms in by 20% to simulate closer viewpoints.<br>
**Horizontal Flip:** Flips images horizontally to simulate images from both left and right views.<br>
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Rescale pixel values from [0, 255] to [0, 1]
    rotation_range=40,            # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,        # Shift images horizontally by up to 20%
    height_shift_range=0.2,       # Shift images vertically by up to 20%
    shear_range=0.2,              # Apply shear transformations
    zoom_range=0.2,               # Apply zoom transformations
    horizontal_flip=True,         # Flip images horizontally
    fill_mode='nearest'           # Fill in missing pixels after transformation
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation
```
---
### 3. Building the CNN Model Architecture:-
Our model is a Convolutional Neural Network (CNN) designed for image classification. CNNs are particularly effective for this task because they can learn spatial hierarchies of features directly from images.<br>

**Model Layers:**<br>
**Input Layer:** Processes 150x150 RGB images.<br>
**Convolutional Layers:** We stack four Conv2D layers, each with ReLU activation to introduce non-linearity. Each convolutional layer is followed by a MaxPooling2D layer, which down-samples the feature maps, reducing computational load and focusing on the most prominent features.<br>
**Flatten Layer:** Converts the 2D output of the convolutional layers into a 1D array, preparing it for the dense layers.<br>
**Dense Layers:** Fully connected layers, with the final layer using softmax activation for classification into the five apple varieties.
#### Compiling and Training Code:
```python
model = models.Sequential()

# First Conv Layer with Max Pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Conv Layer with Max Pooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Conv Layer with Max Pooling
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth Conv Layer with Max Pooling
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer to convert 2D matrix to 1D array
model.add(layers.Flatten())

# Fully connected layer with ReLU
model.add(layers.Dense(512, activation='relu'))

# Output layer with Softmax for classification into 5 classes
model.add(layers.Dense(5, activation='softmax'))
```
---
### 4. Compiling and Training the Model:-
With our architecture in place, we need to compile the model, specifying:

**Loss Function:** Categorical cross-entropy, suited for multi-class classification.<br>
**Optimizer:** Adam, an adaptive learning rate optimizer that works well in image classification.<br>
**Metrics:** Accuracy, to monitor model performance during training.
#### Compiling and Training Code:
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,  # Adjust as needed for training duration
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```
---
### 5. Evaluating the Model:-
After training, it’s crucial to evaluate the model’s performance on the validation dataset to understand its generalization ability. This process reveals how well the model can classify apple varieties it hasn’t seen during training.
```python
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f"Test accuracy: {test_acc}")
```
**Note:** High test accuracy indicates good performance, while a lower score might signal overfitting, suggesting the need for further data augmentation or more training data.

---
### 6. Making Predictions on New Images
With the model trained and evaluated, we’re ready to use it for predictions on unseen images. Here’s the process:

1. Load and preprocess the new image by resizing it to 150x150 pixels and rescaling pixel values.
2. Use the model to predict the class with the highest probability.
3. Print the prediction, indicating the apple variety.
#### Prediction Code:
```python
# Specify the path to a new image
image_path = r'C:\Users\dell\Desktop\download (1).jpeg'

# Load, resize, and preprocess the image
img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

# Map index to class name
predicted_class_name = class_names[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")
```
---
### Conclusion:-
This project demonstrates the power of deep learning in image classification tasks. Our CNN model successfully distinguishes between different apple varieties, a useful application in agriculture and food quality control. The project could be further enhanced by:

1. Using a larger dataset to improve generalization.
2. Implementing transfer learning for more sophisticated feature extraction.
3. Hyperparameter tuning to optimize the model’s performance.
4. With these techniques, you’ll be well on your way to building an even more accurate and versatile image classification model.
