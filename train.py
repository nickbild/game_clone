import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import glob


MSE_THRESHOLD = 0.00000001
CONTINUE_TRAINING_MODEL = ""


class stopCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_loss') <= MSE_THRESHOLD) or os.path.isfile("training.stop"):   
            print("\nReached {0} MSE; stopping training.".format(MSE_THRESHOLD)) 
            self.model.stop_training = True


class CustomLossLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {} # Ensure logs is a dictionary even if empty
        print(f"\n--- Epoch {epoch+1} ---")
        print(f"Total Loss: {logs.get('loss'):.4e} - Val Total Loss: {logs.get('val_loss'):.4e}")

        # Iterate through the logs to find individual loss and metric values.
        for key, value in logs.items():
            if '_loss' in key and not key.startswith('val_'): # Training loss for individual heads
                print(f"  Train {key.replace('_loss', ' Loss')}: {value:.4e}")
            elif key.startswith('val_') and '_loss' in key: # Validation loss for individual heads
                print(f"  Validation {key.replace('val_', '').replace('_loss', ' Loss')}: {value:.4e}")
            
        print("--------------------")


data = []
train_x = []
train_y_paddle1_pos = []
train_y_paddle2_pos = []
train_y_ball_state = []
train_y_game_state = []

BALL_X_MIN, BALL_X_MAX = 19, 781
BALL_Y_MIN, BALL_Y_MAX = 9, 591
PADDLE_Y_MIN, PADDLE_Y_MAX = 60, 540
PADDLE_HALF_HEIGHT = 50

x_range = BALL_X_MAX - BALL_X_MIN
y_range = BALL_Y_MAX - BALL_Y_MIN


print ("Reading in game data ({0})...".format(datetime.datetime.now()))

files1 = glob.glob("data/game_data_*.txt")
files2 = glob.glob("data_sim/pong_training_data_*.txt")
all_files = files1 + files2

for filename in all_files:
    # Read the game data into memory.
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            pieces = line.strip().split()
            paddle1_pos = int(pieces[0])
            paddle2_pos = int(pieces[1])
            ball_x = int(pieces[2])
            ball_y = int(pieces[3])
            paddle1_vel = int(pieces[4])
            paddle2_vel = int(pieces[5])
            game_state = int(pieces[6])
            data.append([paddle1_pos, paddle2_pos, ball_x, ball_y, paddle1_vel, paddle2_vel, game_state])        

    # Create the training data structure.
    step = 1
    # if filename.endswith("game_data_100.txt"):
    #     step = 5
    
    for pos in range(0, len(data)-4, step):
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

        paddle1_pos_4 = data[pos+3][0] 
        paddle2_pos_4 = data[pos+3][1]
        ball_x_4 = data[pos+3][2]
        ball_y_4 = data[pos+3][3]
        paddle1_vel_4 = data[pos+3][4]
        paddle2_vel_4 = data[pos+3][5]
        
        paddle1_pos_5 = data[pos+4][0] 
        paddle2_pos_5 = data[pos+4][1]
        ball_x_5 = data[pos+4][2]
        ball_y_5 = data[pos+4][3]
        game_state_5 = data[pos+4][6]

        # Compute ball deltas (velocities)
        delta_x_1 = ball_x_2 - ball_x_1
        delta_y_1 = ball_y_2 - ball_y_1
        delta_x_2 = ball_x_3 - ball_x_2
        delta_y_2 = ball_y_3 - ball_y_2
        delta_x_3 = ball_x_4 - ball_x_3
        delta_y_3 = ball_y_4 - ball_y_3
        delta_x_4 = delta_x_3  # Repeat last known delta for frame 4
        delta_y_4 = delta_y_3

        # Compute normalized distances to boundaries and paddle coverages for each frame
        # Frame 1
        dist_left_1 = (ball_x_1 - BALL_X_MIN) / x_range
        dist_right_1 = (BALL_X_MAX - ball_x_1) / x_range
        dist_top_1 = (ball_y_1 - BALL_Y_MIN) / y_range
        dist_bottom_1 = (BALL_Y_MAX - ball_y_1) / y_range
        coverage_p1_1 = abs(ball_y_1 - paddle1_pos_1) / PADDLE_HALF_HEIGHT
        coverage_p2_1 = abs(ball_y_1 - paddle2_pos_1) / PADDLE_HALF_HEIGHT

        # Frame 2
        dist_left_2 = (ball_x_2 - BALL_X_MIN) / x_range
        dist_right_2 = (BALL_X_MAX - ball_x_2) / x_range
        dist_top_2 = (ball_y_2 - BALL_Y_MIN) / y_range
        dist_bottom_2 = (BALL_Y_MAX - ball_y_2) / y_range
        coverage_p1_2 = abs(ball_y_2 - paddle1_pos_2) / PADDLE_HALF_HEIGHT
        coverage_p2_2 = abs(ball_y_2 - paddle2_pos_2) / PADDLE_HALF_HEIGHT

        # Frame 3
        dist_left_3 = (ball_x_3 - BALL_X_MIN) / x_range
        dist_right_3 = (BALL_X_MAX - ball_x_3) / x_range
        dist_top_3 = (ball_y_3 - BALL_Y_MIN) / y_range
        dist_bottom_3 = (BALL_Y_MAX - ball_y_3) / y_range
        coverage_p1_3 = abs(ball_y_3 - paddle1_pos_3) / PADDLE_HALF_HEIGHT
        coverage_p2_3 = abs(ball_y_3 - paddle2_pos_3) / PADDLE_HALF_HEIGHT

        # Frame 4
        dist_left_4 = (ball_x_4 - BALL_X_MIN) / x_range
        dist_right_4 = (BALL_X_MAX - ball_x_4) / x_range
        dist_top_4 = (ball_y_4 - BALL_Y_MIN) / y_range
        dist_bottom_4 = (BALL_Y_MAX - ball_y_4) / y_range
        coverage_p1_4 = abs(ball_y_4 - paddle1_pos_4) / PADDLE_HALF_HEIGHT
        coverage_p2_4 = abs(ball_y_4 - paddle2_pos_4) / PADDLE_HALF_HEIGHT

        train_x.append([paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1,
            paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2,
            paddle1_pos_3, paddle2_pos_3, ball_x_3, ball_y_3, paddle1_vel_3, paddle2_vel_3,
            paddle1_pos_4, paddle2_pos_4, ball_x_4, ball_y_4, paddle1_vel_4, paddle2_vel_4,
            delta_x_1, delta_y_1, dist_left_1, dist_right_1, dist_top_1, dist_bottom_1, coverage_p1_1, coverage_p2_1,
            delta_x_2, delta_y_2, dist_left_2, dist_right_2, dist_top_2, dist_bottom_2, coverage_p1_2, coverage_p2_2,
            delta_x_3, delta_y_3, dist_left_3, dist_right_3, dist_top_3, dist_bottom_3, coverage_p1_3, coverage_p2_3,
            delta_x_4, delta_y_4, dist_left_4, dist_right_4, dist_top_4, dist_bottom_4, coverage_p1_4, coverage_p2_4])

        train_y_paddle1_pos.append([paddle1_pos_5])
        train_y_paddle2_pos.append([paddle2_pos_5])
        train_y_ball_state.append([ball_x_5, ball_y_5])
        train_y_game_state.append([game_state_5])

print("Done reading in game data ({0}).".format(datetime.datetime.now()))

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)

else:
    # Build the model.
    main_input = keras.Input(shape=(56,), name='main_input')

    normalization_layer = keras.layers.Normalization(axis=-1, name='input_normalization')
    normalized_input = normalization_layer(main_input)

    # Add some noise to help with model generalization.
    noisy_input = keras.layers.GaussianNoise(stddev=0.01)(normalized_input, training=True)

    # Branch for the paddle1 features.
    paddle1_features = keras.ops.take(noisy_input, indices=[0, 4, 6, 10, 12, 16, 18, 22], axis=1)
    paddle1_branch = keras.layers.Dense(64, activation='relu', name='paddle1_1')(paddle1_features)
    
    # Branch for the paddle2 features.
    paddle2_features = keras.ops.take(noisy_input, indices=[1, 5, 7, 11, 13, 17, 19, 23], axis=1)
    paddle2_branch = keras.layers.Dense(64, activation='relu', name='paddle2_1')(paddle2_features)
    
    # Branch for ball features, including new engineered features.
    ball_indices = [2, 3, 24, 25, 26, 27, 28, 29, 30, 31,  # Frame 1: x, y, dx, dy, dl, dr, dt, db, c1, c2
                    8, 9, 32, 33, 34, 35, 36, 37, 38, 39,  # Frame 2
                    14, 15, 40, 41, 42, 43, 44, 45, 46, 47,  # Frame 3
                    20, 21, 48, 49, 50, 51, 52, 53, 54, 55]  # Frame 4
    ball_features = keras.ops.take(noisy_input, indices=ball_indices, axis=1)
    ball_branch = keras.layers.Reshape((4, 10))(ball_features)
    ball_branch = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(ball_branch, ball_branch)
    ball_branch = keras.layers.LayerNormalization()(ball_branch)
    ball_branch = keras.layers.Flatten()(ball_branch)
    ball_branch = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1), name='ball_1')(ball_branch)
    ball_branch = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1), name='ball_2')(ball_branch)
    
    # Combine ball with paddle features.
    combined_features = keras.layers.Concatenate(name='concatenate_branches')([ball_branch, paddle1_branch, paddle2_branch])
    shared_branch = keras.layers.Reshape((1, -1))(combined_features)
    shared_branch = keras.layers.LSTM(128, return_sequences=False, name='ball_paddle_lstm')(shared_branch)
    shared_branch = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1), name='ball_paddle_dense')(shared_branch)
    shared_branch = keras.layers.Dropout(0.2, name='ball_paddle_dropout')(shared_branch)
    # shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_1')(combined_features)
    # shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_2')(shared_branch)
    # shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_3')(shared_branch)

    paddle1_pos_output = keras.layers.Dense(1, activation='linear', name='paddle1_output_1')(paddle1_branch)
    paddle2_pos_output = keras.layers.Dense(1, activation='linear', name='paddle2_output_1')(paddle2_branch)
    ball_state_output = keras.layers.Dense(2, activation='linear', name='ball_output_2')(shared_branch)
    game_state_output = keras.layers.Dense(1, activation='sigmoid', name='game_state_output_2')(shared_branch)

    model = keras.Model(
        inputs=main_input,
        outputs=[paddle1_pos_output, paddle2_pos_output, ball_state_output, game_state_output],
    )

    # Compile the model.
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=30000,
        decay_rate=0.96
    )
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate_schedule)
    
    model.compile(optimizer=optimizer, 
        loss={
            'paddle1_output_1': 'mse',
            'paddle2_output_1': 'mse',
            'ball_output_2': keras.losses.Huber(delta=3.0),
            'game_state_output_2': 'binary_crossentropy'
        }
    )

# Print the model summary.
model.summary()

# Find edge cases for oversampling.
bounce_indices = []
for i in range(len(train_x)):
    # Extract frame 4 ball position
    ball_x_4, ball_y_4 = train_x[i][20], train_x[i][21]
    ball_x_5, ball_y_5 = train_y_ball_state[i]
    # Check for y-bounce (top/bottom)
    if (ball_y_4 <= BALL_Y_MIN + 3 and ball_y_5 > ball_y_4) or (ball_y_4 >= BALL_Y_MAX - 3 and ball_y_5 < ball_y_4):
        bounce_indices.append(i)
    # Check for x-bounce (paddle hit) or game end
    elif (ball_x_4 <= BALL_X_MIN + 3 and ball_x_5 > ball_x_4) or (ball_x_4 >= BALL_X_MAX - 3 and ball_x_5 < ball_x_4):
        bounce_indices.append(i)
    elif train_y_game_state[i][0] == 1:
        bounce_indices.append(i)

# Oversample: append copies of bounce sequences.
oversample_factor = 6
for idx in bounce_indices:
    for _ in range(oversample_factor - 1):
        train_x.append(train_x[idx])
        train_y_paddle1_pos.append(train_y_paddle1_pos[idx])
        train_y_paddle2_pos.append(train_y_paddle2_pos[idx])
        train_y_ball_state.append(train_y_ball_state[idx])
        train_y_game_state.append(train_y_game_state[idx])

# Prepare the training data.
train_x = np.asarray(train_x).astype('float32')

train_y_paddle1_pos = np.asarray(train_y_paddle1_pos).astype('float32')
train_y_paddle2_pos = np.asarray(train_y_paddle2_pos).astype('float32')
train_y_ball_state = np.asarray(train_y_ball_state).astype('float32')
train_y_game_state = np.asarray(train_y_game_state).astype('float32')

# Adapt the Normalization layer to the training data.
if 'normalization_layer' in locals():
    normalization_layer.adapt(train_x)

# Group together the training labels.
train_y_dict = {
    'paddle1_output_1': train_y_paddle1_pos,
    'paddle2_output_1': train_y_paddle2_pos,
    'ball_output_2': train_y_ball_state,
    'game_state_output_2': train_y_game_state
}

# Shuffle the training data.
# This mixes up the data chosen for the validation set,
# which may otherwise be out of distribution b/c I am trying
# to teach the model some specific behaviors.
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
train_x = train_x[indices]
for key in train_y_dict:
    train_y_dict[key] = train_y_dict[key][indices]

print("Training data shape: {0}".format(train_x.shape))
print("Training labels shape: {0} {1} {2} {3}".format(train_y_paddle1_pos.shape, train_y_paddle2_pos.shape, train_y_ball_state.shape, train_y_game_state.shape))

# Train the model.
custom_logger = CustomLossLogger()
callbacks = stopCallback()
callbacks_list = [stopCallback(), custom_logger]

model.fit(train_x, train_y_dict, batch_size=64, epochs=5000, verbose=2, validation_split=0.2, callbacks=callbacks_list)
model.save("pong.keras")
print("Model training complete!")
