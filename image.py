import tensorflow as tf
import tensorflow.keras.datasets
import random 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# augmentation function
def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = 255 - x

# define the cnn model
model = Sequential([
    # input convolutional layer 
    # convolution operation 
    Conv2D(),
    # reduces the dimensions of the data -- the number of parameters and comp. cost 
    MaxPooling2D(),
    Conv2D(),
    MaxPooling2D(),
    # convert the 2d feature maps into a 1d vector 
    Flatten(),
    # all nodes from the prev layer are connected to all nodes in the current layer
    Dense(),
    Dense()
])

# Data Loading and Preprocessing

# Compile, train and evaluate the model 
model.compile


# comparison 

