# Classifying Apple Varieties

**Classifying Apple Varieties with Deep Learning: A Step-by-Step Guide**

**Introduction**<br>
Fruit classification is a great application of computer vision, often used in agriculture and retail industries to sort and grade produce. In this post, we’ll build a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify different varieties of apples based on their images. Specifically, we’ll classify five apple types: **Fuji, Golden Delicious, Granny Smith, Lady, and Red Delicious.**

**Dataset Preparation**<br>
The dataset is organized into training and validation sets, each with folders for the five apple varieties. The training and validation paths are:

*Training directory: C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Train
Validation directory: C:/Users/dell/Desktop/Minor Project/APPLE VARIETIES IMAGE DATASET/Validation*

**Data Augmentation**<br>
To make our model more robust, we’ll use data augmentation techniques to create modified versions of our images. This includes random rotations, width/height shifts, shearing, zooming, and horizontal flips. Here’s how we set it up:

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

**Model Architecture**<br>
Our CNN model comprises convolutional and max-pooling layers, followed by fully connected layers. Here’s the model structure:

*Input layer: Accepts 150x150 RGB images.
Convolutional layers: We use four Conv2D layers to extract features, each followed by MaxPooling2D layers to reduce spatial dimensions.
Flatten layer: Converts the 2D output to a 1D array for the dense layers.
Dense layers: Two dense layers with the final layer using a softmax activation function to classify the images into five classes.*

**Compiling and Training the Model**<br>
We use categorical cross-entropy as the loss function, Adam optimizer, and accuracy as the evaluation metric. The model is trained for 15 epochs. Here’s how we compiled and trained it:

*model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)*

**Evaluating the Model**<br>
After training, we evaluate the model’s accuracy on the validation dataset:

*test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f"Test accuracy: {test_acc}")*

**Making Predictions**<br>
To test a new image, we preprocess it similarly and pass it to the model to get a prediction.

*img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)
predicted_class_name = class_names[np.argmax(predictions[0])]
print(f"Predicted class: {predicted_class_name}")*

**Conclusion**<br>
This project shows the potential of deep learning for image classification tasks. The model’s accuracy can be further improved by fine-tuning parameters, using a larger dataset, or implementing transfer learning.
