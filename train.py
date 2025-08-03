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
        print(f"Total Loss: {logs.get('loss'):.8f} - Val Total Loss: {logs.get('val_loss'):.8f}")

        # Iterate through the logs to find individual loss and metric values
        # Keras names these automatically: 'output_name_loss', 'val_output_name_loss', 'output_name_metric', etc.
        for key, value in logs.items():
            if '_loss' in key and not key.startswith('val_'): # Training loss for individual heads
                print(f"  Train {key.replace('_loss', ' Loss')}: {value:.8f}")
            elif key.startswith('val_') and '_loss' in key: # Validation loss for individual heads
                print(f"  Validation {key.replace('val_', '').replace('_loss', ' Loss')}: {value:.8f}")
            elif '_accuracy' in key and not key.startswith('val_'): # Training accuracy for individual heads
                print(f"  Train {key.replace('_accuracy', ' Accuracy')}: {value:.8f}")
            elif key.startswith('val_') and '_accuracy' in key: # Validation accuracy for individual heads
                print(f"  Validation {key.replace('val_', '').replace('_accuracy', ' Accuracy')}: {value:.8f}")
            elif '_mae' in key and not key.startswith('val_'): # Training MAE for individual heads
                print(f"  Train {key.replace('_mae', ' MAE')}: {value:.8f}")
            elif key.startswith('val_') and '_mae' in key: # Validation MAE for individual heads
                print(f"  Validation {key.replace('val_', '').replace('_mae', ' MAE')}: {value:.8f}")

        print("--------------------")


data = []
train_x = []
train_y_paddle1_pos = []
train_y_paddle2_pos = []
train_y_ball_state = []
train_y_game_state = []


print ("Reading in game data ({0})...".format(datetime.datetime.now()))

for filename in glob.glob("data/game_data_*.txt"):
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
    if filename.endswith("game_data_100.txt"):
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
        game_state_3 = data[pos+2][6]
                
        train_x.append([paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1, 
            paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2])

        train_y_paddle1_pos.append([paddle1_pos_3])
        train_y_paddle2_pos.append([paddle2_pos_3])
        train_y_ball_state.append([ball_x_3, ball_y_3])
        train_y_game_state.append([game_state_3])

print("Done reading in game data ({0}).".format(datetime.datetime.now()))

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)

else:
    # Build the model.
    main_input = keras.Input(shape=(12,), name='main_input')
    
    # Branch for the paddle1 features.
    paddle1_features = keras.ops.take(main_input, indices=[0, 4, 6, 10], axis=1)
    paddle1_branch = keras.layers.Dense(192, activation='relu', name='paddle1_1')(paddle1_features)
    paddle1_branch = keras.layers.Dense(192, activation='relu', name='paddle1_2')(paddle1_branch)
    paddle1_branch = keras.layers.Dense(192, activation='relu', name='paddle1_3')(paddle1_branch)
    paddle1_branch = keras.layers.Dense(192, activation='relu', name='paddle1_4')(paddle1_branch)
    paddle1_branch = keras.layers.Dense(192, activation='relu', name='paddle1_5')(paddle1_branch)
    
    # Branch for the paddle2 features.
    paddle2_features = keras.ops.take(main_input, indices=[1, 5, 7, 11], axis=1)
    paddle2_branch = keras.layers.Dense(192, activation='relu', name='paddle2_1')(paddle2_features)
    paddle2_branch = keras.layers.Dense(192, activation='relu', name='paddle2_2')(paddle2_branch)
    paddle2_branch = keras.layers.Dense(192, activation='relu', name='paddle2_3')(paddle2_branch)
    paddle2_branch = keras.layers.Dense(192, activation='relu', name='paddle2_4')(paddle2_branch)
    paddle2_branch = keras.layers.Dense(192, activation='relu', name='paddle2_5')(paddle2_branch)

    # Branch for ball features.
    ball_features = keras.ops.take(main_input, indices=[2, 3, 8, 9], axis=1)
    ball_branch = keras.layers.Dense(192, activation='relu', name='ball_1')(ball_features)
    ball_branch = keras.layers.Dense(192, activation='relu', name='ball_2')(ball_branch)
    ball_branch = keras.layers.Dense(192, activation='relu', name='ball_3')(ball_branch)
    ball_branch = keras.layers.Dense(192, activation='relu', name='ball_4')(ball_branch)
    ball_branch = keras.layers.Dense(192, activation='relu', name='ball_5')(ball_branch)

    # Combine ball with paddle positions.
    combined_features = keras.layers.Concatenate(name='concatenate_branches')([ball_branch, paddle1_branch, paddle2_branch])
    shared_branch = keras.layers.Dense(192, activation='relu', name='ball_paddle_1')(combined_features)
    shared_branch = keras.layers.Dense(192, activation='relu', name='ball_paddle_2')(shared_branch)
    shared_branch = keras.layers.Dense(192, activation='relu', name='ball_paddle_3')(shared_branch)
    shared_branch = keras.layers.Dense(192, activation='relu', name='ball_paddle_4')(shared_branch)

    # Output heads.
    paddle1_output_head = keras.layers.Dense(1, activation='linear', name='paddle1_6')(paddle1_branch)
    paddle2_output_head = keras.layers.Dense(1, activation='linear', name='paddle2_6')(paddle2_branch)

    ball_output_head = keras.layers.Dense(192, activation='relu', name='ball_output_1')(shared_branch)
    ball_output_head = keras.layers.Dense(2, activation='linear', name='ball_output_2')(ball_output_head)

    game_state_output_head = keras.layers.Dense(192, activation='relu', name='game_state_output_1')(shared_branch)
    game_state_output_head = keras.layers.Dense(1, activation='sigmoid', name='game_state_output_2')(game_state_output_head)

    model = keras.Model(inputs=main_input, outputs=[paddle1_output_head, paddle2_output_head, ball_output_head, game_state_output_head])

    # Compile the model.
    learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=12000,
        decay_rate=0.96
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    
    model.compile(optimizer=optimizer, 
        loss={
            'paddle1_6': 'mse',
            'paddle2_6': 'mse',
            'ball_output_2': 'mse',
            'game_state_output_2': 'binary_crossentropy'
        }
    )

# Print the model summary.
model.summary()

# Prepare the training data.
train_x = np.asarray(train_x).astype('float32')

train_y_paddle1_pos = np.asarray(train_y_paddle1_pos).astype('float32')
train_y_paddle2_pos = np.asarray(train_y_paddle2_pos).astype('float32')
train_y_ball_state = np.asarray(train_y_ball_state).astype('float32')
train_y_game_state = np.asarray(train_y_game_state).astype('float32')

# Scale the training data to between 0 and 1.
x_min = np.min(train_x, axis=0)
x_max = np.max(train_x, axis=0)
x_range = x_max - x_min
x_range[x_range == 0] = 1.0 # Prevent division by zero for constant features

y_paddle1_min = np.min(train_y_paddle1_pos, axis=0)
y_paddle1_max = np.max(train_y_paddle1_pos, axis=0)
y_paddle1_range = y_paddle1_max - y_paddle1_min
y_paddle1_range[y_paddle1_range == 0] = 1.0

y_paddle2_min = np.min(train_y_paddle2_pos, axis=0)
y_paddle2_max = np.max(train_y_paddle2_pos, axis=0)
y_paddle2_range = y_paddle2_max - y_paddle2_min
y_paddle2_range[y_paddle2_range == 0] = 1.0

y_ball_min = np.min(train_y_ball_state, axis=0)
y_ball_max = np.max(train_y_ball_state, axis=0)
y_ball_range = y_ball_max - y_ball_min
y_ball_range[y_ball_range == 0] = 1.0

# Apply scaling
train_x = (train_x - x_min) / x_range
train_y_paddle1_pos = (train_y_paddle1_pos - y_paddle1_min) / y_paddle1_range
train_y_paddle2_pos = (train_y_paddle2_pos - y_paddle2_min) / y_paddle2_range
train_y_ball_state = (train_y_ball_state - y_ball_min) / y_ball_range

# Group together the training labels.
train_y_dict = {
    'paddle1_6': train_y_paddle1_pos,
    'paddle2_6': train_y_paddle2_pos,
    'ball_output_2': train_y_ball_state,
    'game_state_output_2': train_y_game_state
}

# Shuffle the training data.
# This mixes up the data chosen for the validation set,
# which may otherwise be out of distribution b/c I am trying
# to teach the model some specific behaviors.
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
# Apply the shuffled indices to the data.
tmp = train_x[indices]
train_x = tmp
tmp = []

for key in train_y_dict:
    train_y_dict[key] = train_y_dict[key][indices]

print("Training data shape: {0}".format(train_x.shape))
print("Training labels shape: {0} {1} {2} {3}".format(train_y_paddle1_pos.shape, train_y_paddle2_pos.shape, train_y_ball_state.shape, train_y_game_state.shape))

# Train the model.
custom_logger = CustomLossLogger()
callbacks = stopCallback()
callbacks_list = [stopCallback(), custom_logger]

model.fit(train_x, train_y_dict, batch_size=32, epochs=5000, verbose=2, validation_split=0.2, callbacks=callbacks_list)
model.save("pong.keras")
print("Model training complete!")

# Save data scaling information.
np.save('x_min.npy', x_min)
np.save('x_max.npy', x_max)
np.save('y_paddle1_min.npy', y_paddle1_min)
np.save('y_paddle1_max.npy', y_paddle1_max)
np.save('y_paddle2_min.npy', y_paddle2_min)
np.save('y_paddle2_max.npy', y_paddle2_max)
np.save('y_ball_min.npy', y_ball_min)
np.save('y_ball_max.npy', y_ball_max)
print("Metadata saved.")
