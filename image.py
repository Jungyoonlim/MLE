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
    
])
model.compile


# training and validation wo augmentation 

# comparison 

