import tensorflow
from tensorflow import keras
from PIL import Image
import glob
import numpy as np
import os


MAE_THRESHOLD = 0.0001


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
        if(logs.get('mae') < MAE_THRESHOLD):   
            print("\nReached %2.2f% MAE; stopping training." %(MAE_THRESHOLD))   
            self.model.stop_training = True


train_x = []
train_y = []

print ("Reading in image data...")

# Get all sets of files (1 input, 1 expected output).
files = glob.glob("img/screen-*.jpg")
times = {}
for f in files:
    times[f.split("-")[1]] = True

# Inspect each set.
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
                for r in range(10):
                    inp.append(btn)
                train_x.append(inp)
            else:
                train_y.append(inp)

            f = open(cache, "w")
            f.write(",".join(str(i) for i in inp))
            f.close()

    # break

print("Done reading in image data.")

# Build the model.
input_size = 19210
output_size = 19200

model = keras.Sequential([
    keras.layers.Input(shape=(input_size,)),
    keras.layers.Dense(units=800, activation='leaky_relu'),
    keras.layers.Dense(units=400, activation='leaky_relu'),
    keras.layers.Dense(units=400, activation='leaky_relu'),
    keras.layers.Dense(units=350, activation='leaky_relu'),
    keras.layers.Dense(units=output_size)
])

# Compile the model.
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

# Print the model summary.
model.summary()

# Train the model.
train_x = np.asarray(train_x).astype('float32')
train_y = np.asarray(train_y).astype('float32')

callbacks = stopCallback()
model.fit(train_x, train_y, batch_size=64, epochs=500, verbose=2, validation_split=0.2, callbacks=[callbacks])
model.save("pong.keras")

# model = keras.models.load_model("pong.keras")
# # model.evaluate(test_x ,test_y)
# res = model(np.expand_dims(train_x[0], axis=0), training=False)
# # print(res)
# np.savetxt('test.txt', res, fmt='%d')
