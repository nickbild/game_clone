import tensorflow
from tensorflow import keras
import numpy as np
import sys


in_sample_num = int(sys.argv[1])

train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

model = keras.models.load_model("pong.keras")

print(train_x[in_sample_num])
new_prediction = model.predict(np.expand_dims(train_x[in_sample_num], axis=0))
print(train_y[in_sample_num])
print(new_prediction)
