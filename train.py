import tensorflow
from tensorflow import keras
import numpy as np
import os
import datetime


MSE_THRESHOLD = 3
LOAD_SAVED_DATA = False


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


train_x = []
train_y = []
data = []

if not LOAD_SAVED_DATA:
    print ("Reading in game data ({0})...".format(datetime.datetime.now()))

    # Read the game data into memory.
    with open("game_data.txt", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            paddle1_pos, paddle2_pos, ball_x, ball_y, paddle1_vel, paddle2_vel = map(int, line.split())
            data.append([paddle1_pos, paddle2_pos, ball_x, ball_y, paddle1_vel, paddle2_vel])        

    # Create the training data structure.
    for pos in range(len(data)-2):
        paddle1_pos_1 = data[pos][0] 
        paddle2_pos_1 = data[pos][1]
        ball_x_1 = data[pos][2]
        ball_y_1 = data[pos][3]
        paddle1_vel_1 = data[pos][4]
        paddle2_vel_1 = data[pos][5]

        paddle1_pos_2 = data[pos+1][0] 
        paddle2_pos_2 = data[pos+1][1]
        ball_x_2 = data[pos+1][2]
        ball_y_2 = data[pos+1][3]
        paddle1_vel_2 = data[pos+1][4]
        paddle2_vel_2 = data[pos+1][5]
        
        paddle1_pos_3 = data[pos+2][0] 
        paddle2_pos_3 = data[pos+2][1]
        ball_x_3 = data[pos+2][2]
        ball_y_3 = data[pos+2][3]
        paddle1_vel_3 = data[pos+2][4]
        paddle2_vel_3 = data[pos+2][5]

        train_x.append([[paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1], [paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2]])
        train_y.append([paddle1_pos_3, paddle2_pos_3, ball_x_3, ball_y_3])

    data = [] # Free up memory.
    
    print("Done reading in game data ({0}).".format(datetime.datetime.now()))

# Build the model.
model = keras.Sequential()
model.add(keras.layers.LSTM(96, activation="relu", input_shape=(2, 6,)))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(4))

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
    np.save('train_x.npy', train_x)
    np.save('train_y.npy', train_y)
else:
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')

# Train the model.
callbacks = stopCallback()
model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=2, validation_split=0.2, callbacks=[callbacks])
model.save("pong.keras")
print("Model training complete!")
