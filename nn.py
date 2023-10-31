from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# classification of grayscale 28*28 imgs into 10 categories 
model = Sequential([
    Conv2D(28,28,1),
    # pooling to reduce spatial dims and retain the most impt info
    Flatten()
    Dense()
])