import tensorflow
from tensorflow import keras
import numpy as np
import os
import datetime
import glob


MSE_THRESHOLD = 0.5
CONTINUE_TRAINING_MODEL = "pong.2025-07-08_2.keras"


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


train_x = []
train_y = []
data = []


print ("Reading in game data ({0})...".format(datetime.datetime.now()))

# Read the game data into memory.
for filename in glob.glob("game_data_*.txt"):
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            paddle1_pos, paddle2_pos, ball_x, ball_y, paddle1_vel, paddle2_vel = map(int, line.split())
            data.append([paddle1_pos, paddle2_pos, ball_x, ball_y, paddle1_vel, paddle2_vel])        

    # Create the training data structure.
    step = 1
    if filename.endswith("game_data_10.txt"):
        step = 3

    for pos in range(0, len(data)-2, step):
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

        # Ignore breaks in data collection that introduce artificially large changes in screen element positions.
        if abs(paddle1_pos_1 - paddle1_pos_2) > 8 or abs(paddle1_pos_2 - paddle1_pos_3) > 8:
            continue
        if abs(paddle2_pos_1 - paddle2_pos_2) > 8 or abs(paddle2_pos_2 - paddle2_pos_3) > 8:
            continue

        # LSTM:
        train_x.append([[paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1], [paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2]])
        # Dense:
        # train_x.append([paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1, paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2])
        train_y.append([paddle1_pos_3, paddle2_pos_3, ball_x_3, ball_y_3])

    data = [] # Free up memory.

print("Done reading in game data ({0}).".format(datetime.datetime.now()))

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)

else:
    # Build the model.
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(2, 6,)))
    model.add(keras.layers.LSTM(160, activation='relu'))

    # model.add(keras.layers.LSTM(32, return_sequences=True, activation="relu"))
    # model.add(keras.layers.LSTM(32, activation="relu"))
    # model.add(keras.layers.Dense(64, activation="relu"))

    # model.add(keras.layers.Conv1D(128, 2, activation='relu', padding='same'))
    # model.add(keras.layers.MaxPooling1D(2))
    # model.add(keras.layers.Conv1D(128, 2, activation='relu', padding='same'))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.MaxPooling1D(2, padding="same"))
    # model.add(keras.layers.LSTM(96, return_sequences=True, activation="relu"))
    # model.add(keras.layers.LSTM(64, activation="relu"))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(4))

    # Compile the model.
    optimizer = keras.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer,
                loss='mse',
                metrics=['mse', 'mae'])

# Print the model summary.
model.summary()

# Prepare the training data.
train_x = np.asarray(train_x).astype('float32')
train_y = np.asarray(train_y).astype('float32')

# Shuffle the training data.
# This mixes up the data chosen for the validation set,
# which may otherwise be out of distribution b/c I am trying
# to teach the model some specific behaviors.
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
# Apply the shuffled indices to the data.
tmp = train_x[indices]
train_x = tmp
tmp = train_y[indices]
train_y = tmp
tmp = []

print("Training data shape: {0}".format(train_x.shape))
print("Training labels shape: {0}".format(train_y.shape))

# Train the model.
callbacks = stopCallback()
model.fit(train_x, train_y, batch_size=8, epochs=5000, verbose=2, validation_split=0.2, callbacks=[callbacks])
model.save("pong.keras")
print("Model training complete!")
