import tensorflow
from tensorflow import keras
import numpy as np
import time
import os


in_sample_num = 38

train_x = np.load('train_x.npy')
# train_y = np.load('train_y.npy')

train_x = np.reshape(train_x, (len(train_x), 5, 121, 160, 1))
# train_y = np.reshape(train_y, (len(train_y), 5, 121, 160, 1))

model = keras.models.load_model("pong.keras")

new_prediction = model.predict(np.expand_dims(train_x[in_sample_num], axis=0))


for f in range(5):
    time.sleep(2)
    os.system('clear')
    print("++++ ", end='')
    print(train_x[in_sample_num][f][120][0])
    for r in range(120):
        for c in range(160):
            v = new_prediction[0][f][r][c]
            if v > 0.5:
                v = "*"
            else:
                v = " "

            print(str(v) + "", end='')
        print()
    print("----")
