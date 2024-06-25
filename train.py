import tensorflow
from tensorflow import keras
from PIL import Image
import glob
import numpy as np
import os
import datetime


MSE_THRESHOLD = 0.00065
LOAD_SAVED_DATA = True
TEST_MODE = False
TEST_SAMPLES = 5


def read_data(img_data):
    lst = []
    for y in range(120):
        for x in range(160):
            r = img_data[x, y][0]
            
            if r < 100:
                r = 0
            else:
                r = 1

            lst.append(r)

    return lst


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


train_x = []
train_y = []

if not LOAD_SAVED_DATA:
    if TEST_MODE:
        print("**** TEST MODE ON! ****")
    print ("Reading in image data ({0})...".format(datetime.datetime.now()))

    # Get all sets of files (1 input, 1 expected output).
    files = glob.glob("img/screen-*.jpg")
    times = {}
    for f in files:
        times[f.split("-")[1]] = True

    # Inspect each set.
    cnt = 0
    for t in times:
        inp_x = []
        inp_y = []
        tmp = []

        for idx in range(5):
            files_x = glob.glob("img/screen-{0}-{1}-*.jpg".format(t, idx))
            files_y = glob.glob("img/screen-{0}-{1}-*.jpg".format(t, idx+1))

            img_x = Image.open(files_x[0])
            img_data_x = img_x.load()

            img_y = Image.open(files_y[0])
            img_data_y = img_y.load()

            # Get button state.
            btn = files_x[0].split("-")[3].replace(".jpg", "")

            tmp = read_data(img_data_x)
            for _ in range(160):
                tmp.append(btn)
            inp_x.append(tmp)

            tmp = read_data(img_data_y)
            for _ in range(160):
                tmp.append(0)
            inp_y.append(tmp)

        train_x.append(inp_x)    
        train_y.append(inp_y)

        if TEST_MODE:
            cnt += 1
            if cnt > TEST_SAMPLES:
                break

    print("Done reading in image data ({0}).".format(datetime.datetime.now()))

# Build the model.
model = keras.Sequential()
model.add(keras.layers.Input((5, 121, 160, 1)))
model.add(keras.layers.ConvLSTM2D(filters=24, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(2, 2), padding="same", return_sequences=True, activation="relu"))
model.add(keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"))

# Compile the model.
optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mse', 'mae'])

# Print the model summary.
model.summary()

if not LOAD_SAVED_DATA:
    train_x = np.asarray(train_x).astype('float32')
    train_y = np.asarray(train_y).astype('float32')
    if not TEST_MODE:
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
else:
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')

# Reshape for the ConvLSTM input.
train_x = np.reshape(train_x, (len(train_x), 5, 121, 160, 1))
train_y = np.reshape(train_y, (len(train_y), 5, 121, 160, 1))

# Train the model.
callbacks = stopCallback()
model.fit(train_x, train_y, batch_size=16, epochs=100, verbose=2, validation_split=0.2, callbacks=[callbacks])
model.save("pong.keras")
print("Model training complete!")
