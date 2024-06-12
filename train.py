import tensorflow
from tensorflow import keras
from PIL import Image
import glob
import numpy as np


train_x = []
train_y = []

print ("Reading in image data...")

# Get all sets of files (3 input, 1 expected output).
files = glob.glob("img/screen-*.jpg")
times = {}
for f in files:
    times[f.split("-")[1]] = True

# Inspect each set.
for time in times:
    inp = []
    out = []

    # Concatenate the inputs.
    for idx in range(3):
        files = glob.glob("img/screen-{0}-{1}-*.jpg".format(time, idx))
        img = Image.open(files[0])
        img_data = img.load()

        # Get button state.
        btn = files[0].split("-")[3].replace(".jpg", "")

        for y in range(120):
            for x in range(160):
                r = img_data[x, y][0]
                
                if r < 100:
                    r = 0
                elif r < 150:
                    r = 1 #130
                else:
                    r = 2 #234

                inp.append(r)

        inp.append(btn)

    train_x.append(inp)

    # Prepare the outputs.
    files = glob.glob("img/screen-{0}-3-*.jpg".format(time))
    img = Image.open(files[0])
    img_data = img.load()

    # Get button state.
    btn = int(files[0].split("-")[3].replace(".jpg", ""))

    for y in range(120):
        for x in range(160):
            r = img_data[x, y][0]
            
            if r < 100:
                r = 0
            elif r < 150:
                r = 1 #130
            else:
                r = 2 #234

            out.append(r)
    
    train_y.append(out)

print("Done reading in image data.")

# Build the model.
input_size = 57603
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
              loss='mse',
              metrics=['accuracy'])

# Print the model summary.
model.summary()

# Train the model.
train_x = np.asarray(train_x).astype('float32')
train_y = np.asarray(train_y).astype('float32')

model.fit(train_x, train_y, batch_size=64, epochs=50, verbose=2, validation_split=0.2)
model.save("pong.keras")

#model = keras.models.load_model("pong.keras")
# model.evaluate(test_x ,test_y)
#res = model(np.expand_dims(train_x[99], axis=0), training=False)
print(res)
#np.savetxt('test.txt', res, fmt='%d')
