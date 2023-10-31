import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

# data of 5% young ppl who did experience side effects and old ppl wo side effects
for i in range(50):
    young_random = randint(13,64)
    train_samples.append(young_random)
    train_labels.append(1)

    old_random = randint(65,100)
    train_samples.append(old_random)
    train_labels.append(0)

# opp
for i in range(1000):
    young_random = randint(13,64)
    train_samples.append(young_random)
    train_labels.append(0)

    old_random = randint(65,100)
    train_samples.append(old_random)
    train_labels.append(1)

# data transformation
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# further processing! standardize and normalize 
# 13 to 100 --> 0 to 1 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

'''
If running a GPU:
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

# Sequential Model; linear stack of layers 
model = Sequential([
    # First hidden layer, 16 neurons 
    # relu: if x <= 0 --> 0, x > 0 --> x
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    # patients had side-effect or not. 
    Dense(units=2, activation='softmax')
])

model.summary()

# compile function prepares the model for training 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# batch size --> the # of samples that will be propagated through the network
# epocs --> number of times it will iterate thru the data 
model.fit(x=scaled_train_samples, y=train_labels,batch_size=10, epocs=30,shuffle=True, verbose=2)

# validation set -- how our model is doing! 

