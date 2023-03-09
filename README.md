# CNN using Keras-TensorFlow on CIFAR-10 dataset

This project involves building three Convolutional Neural Network (CNN) models using Keras-TensorFlow on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class.
The goal of the project is to compare the performance of each model and determine the optimal configuration for the CIFAR-10 dataset.
## Getting Started

To get started with this project, you will need to install the following packages:

- Tensorflow
- Keras
- Matplotlib

You can install these packages using the following command:
``` bash
pip install tensorflow keras matplotlib
```

## Dataset
The CIFAR-10 dataset will be downloaded and loaded into the project using the following code:
``` python
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```

## Preprocessing 
The pixel values of the images will be normalized to be between 0 and 1 using the following code:
``` python
train_images, test_images = train_images / 255.0 , test_images / 255.0
```

## Models
### A-Model 1
The first model has 1 hidden layer, ReLU activation function, and SGD optimizer.It is trained for 10 epochs with a batch size of 64.
=> loss: 1.5534 - accuracy: 0.5539
### B-Model 2
The second model has 3 hidden layers, ReLU activation function, and SGD optimizer. It is trained for 15 epochs with a batch size of 64.
=> loss: 0.9220 - accuracy: 0.6893
### C-Model 3
The third model has 3 hidden layers, tanh activation function, and SGD optimizer. It is trained for 15 epochs with a batch size of 64.
=> loss: 1.4131 - accuracy: 0.5850
### D-Model 4
The first model has 3 hidden layer, ReLU activation function, and ADAM optimizer. It is trained for 10 epochs with a batch size of 64.
=> loss: 1.0308 - accuracy: 0.6815
### E-Model 5
The first model has 3 hidden layer, ReLU activation function, and ADAM optimizer and also a Dropout. It is trained for 10 epochs with a batch size of 64.
=> loss: 0.9923 - accuracy: 0.6743

## Run
To run the code, first, import `TensorFlow` and load the `CIFAR-10` dataset using the datasets module. The pixel values are normalized between 0 and 1, and the class names are defined. Then, 25 images from the training set are displayed using matplotlib.

Each model is defined using the `models.Sequential()` method. The layers are added using `model.add()`. After defining the layers, the model is compiled with the optimizer, loss function, and metrics using `model.compile()`. Finally, the model is trained using `model.fit()` and evaluated using `model.evaluate()`. The training and validation accuracy and loss are plotted using matplotlib.

Note that the models can be further optimized by changing the hyperparameters such as the learning rate, number of epochs, and batch size. Also, the architecture of the models can be modified by adding or removing layers, changing the number of filters in each layer, and changing the activation functions.

## Conclusion
The best model is the one with the highest accuracy and the lowest loss. The results suggest that Model 2 has the highest accuracy and lowest loss. Further optimization of hyperparameters may improve the performance of the models.
