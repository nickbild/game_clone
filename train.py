import tensorflow
from tensorflow import keras
from PIL import Image
import glob
import numpy as np
import os
import datetime


MSE_THRESHOLD = 0.0001
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
        if(logs.get('val_loss') <= MSE_THRESHOLD):   
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
        for idx in range(2):
            files = glob.glob("img/screen-{0}-{1}-*.jpg".format(t, idx))
            cache = "img.cache/" + files[0].split("/")[-1] + ".txt"

            if os.path.isfile(cache):
                f = open(cache, "r")
                if idx == 0:
                    inp = f.readline().split(',')
                    train_x.append(inp)
                else:
                    inp = f.readline().split(',')
                    train_y.append(inp)
                f.close()

            else:
                inp = []

                img = Image.open(files[0])
                img_data = img.load()

                # Get button state.
                btn = files[0].split("-")[3].replace(".jpg", "")

                inp = read_data(img_data)
                
                if idx == 0:
                    for r in range(160):
                        inp.append(btn)
                    train_x.append(inp)
                else:
                    train_y.append(inp)

                f = open(cache, "w")
                f.write(",".join(str(i) for i in inp))
                f.close()

        if TEST_MODE:
            cnt += 1
            if cnt > TEST_SAMPLES:
                break

    print("Done reading in image data ({0}).".format(datetime.datetime.now()))

# Build the model.
input_size = 19360
output_size = 19200

# model = keras.Sequential([
#     keras.layers.Input(shape=(input_size,)),
#     keras.layers.Dense(units=800, activation='relu'),
#     keras.layers.Dense(units=600, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=400, activation='relu'),
#     keras.layers.Dense(units=output_size)
# ])

model = keras.Sequential()
model.add(keras.layers.Input((160, 121, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(24, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(800, activation='relu'))
model.add(keras.layers.Dense(output_size))

# Compile the model.
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse', 'mae'])

# Print the model summary.
model.summary()

if not LOAD_SAVED_DATA:
    train_x = np.asarray(train_x).astype('float32')
    train_y = np.asarray(train_y).astype('float32')
    if not TEST_MODE:
        np.savetxt('train_x.txt', train_x, fmt='%d')
        np.savetxt('train_y.txt', train_y, fmt='%d')
else:
    train_x = np.loadtxt('train_x.txt', dtype=float)
    train_y = np.loadtxt('train_y.txt', dtype=float)

# Reshape for the CNN input.
train_x = np.reshape(train_x, (len(train_x), 160, 121, 1))

# Train the model.
callbacks = stopCallback()
model.fit(train_x, train_y, batch_size=32, epochs=500, verbose=2, validation_split=0.2, callbacks=[callbacks])
model.save("pong.keras")

# model = keras.models.load_model("pong.keras")
# # model.evaluate(test_x ,test_y)
# res = model(np.expand_dims(train_x[0], axis=0), training=False)
# # print(res)
# np.savetxt('test.txt', res, fmt='%d')
