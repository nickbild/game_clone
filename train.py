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

        train_x.append([paddle1_pos_1, paddle2_pos_1, ball_x_1, ball_y_1, paddle1_vel_1, paddle2_vel_1,
            paddle1_pos_2, paddle2_pos_2, ball_x_2, ball_y_2, paddle1_vel_2, paddle2_vel_2,
            paddle1_pos_3, paddle2_pos_3, ball_x_3, ball_y_3, paddle1_vel_3, paddle2_vel_3,
            paddle1_pos_4, paddle2_pos_4, ball_x_4, ball_y_4, paddle1_vel_4, paddle2_vel_4,])

        train_y_paddle1_pos.append([paddle1_pos_5])
        train_y_paddle2_pos.append([paddle2_pos_5])
        train_y_ball_state.append([ball_x_5, ball_y_5])
        train_y_game_state.append([game_state_5])

print("Done reading in game data ({0}).".format(datetime.datetime.now()))

if CONTINUE_TRAINING_MODEL != "":
    print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
    model = keras.models.load_model(CONTINUE_TRAINING_MODEL)

else:
    # # Build the model.
    # main_input = keras.Input(shape=(24,), name='main_input')
    
    # # Branch for the paddle1 features.
    # paddle1_features = keras.ops.take(main_input, indices=[0, 4, 6, 10, 12, 16, 18, 22], axis=1)
    # paddle1_branch = keras.layers.Dense(32, activation='relu', name='paddle1_1')(paddle1_features)
    # # paddle1_branch = keras.layers.Dense(32, activation='relu', name='paddle1_2')(paddle1_branch)
    
    # # Branch for the paddle2 features.
    # paddle2_features = keras.ops.take(main_input, indices=[1, 5, 7, 11, 13, 17, 19, 23], axis=1)
    # paddle2_branch = keras.layers.Dense(32, activation='relu', name='paddle2_1')(paddle2_features)
    # # paddle2_branch = keras.layers.Dense(32, activation='relu', name='paddle2_2')(paddle2_branch)
    
    # # Branch for ball features.
    # ball_features = keras.ops.take(main_input, indices=[2, 3, 8, 9, 14, 15, 20, 21], axis=1)
    # # reshaped_ball_features = keras.layers.Reshape((4, 2))(ball_features)
    # # ball_branch = keras.layers.LSTM(64)(reshaped_ball_features)
    # ball_branch = keras.layers.Dense(32, activation='relu', name='ball_1')(ball_features)
    # ball_branch = keras.layers.Dense(32, activation='relu', name='ball_2')(ball_branch)
    # # ball_branch = keras.layers.Dense(32, activation='relu', name='ball_3')(ball_branch)
    
    # # Combine ball with paddle features.
    # combined_features = keras.layers.Concatenate(name='concatenate_branches')([ball_branch, paddle1_branch, paddle2_branch])
    # shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_1')(combined_features)
    # shared_branch = keras.layers.Dense(32, activation='relu', name='ball_paddle_2')(shared_branch)
    # # shared_branch = keras.layers.Dense(32, activation='relu', name='ball_paddle_3')(shared_branch)
    
    # # Output heads.
    # paddle1_output_head = keras.layers.Dense(1, activation='linear', name='paddle1_output_1')(paddle1_branch)
    # paddle2_output_head = keras.layers.Dense(1, activation='linear', name='paddle2_output_1')(paddle2_branch)
    # final_ball_output = keras.layers.Dense(2, activation='linear', name='final_ball_output')(shared_branch)
    # game_state_output_head = keras.layers.Dense(1, activation='sigmoid', name='game_state_output_2')(shared_branch)

    # model = keras.Model(inputs=main_input, outputs=[paddle1_output_head, paddle2_output_head, final_ball_output, game_state_output_head])


    # p1_input = keras.Input(shape=(4, 2), name='paddle1_input')
    # p2_input = keras.Input(shape=(4, 2), name='paddle2_input')
    # ball_input = keras.Input(shape=(4, 2), name='ball_input')

    # # These layers add random noise to the input data during training.
    # # This forces the model to learn to handle slightly imperfect feature values,
    # # making it more robust during inference.
    # # NOISE_STDDEV = 0.01
    # # p1_noisy = keras.layers.GaussianNoise(NOISE_STDDEV)(p1_input)
    # # p2_noisy = keras.layers.GaussianNoise(NOISE_STDDEV)(p2_input)
    # # ball_noisy = keras.layers.GaussianNoise(NOISE_STDDEV)(ball_input)

    # # Input branches.
    # p1_stream = keras.layers.Flatten()(p1_input)
    # p1_stream = keras.layers.Dense(32, name='p1_dense')(p1_stream)

    # p2_stream = keras.layers.Flatten()(p2_input)
    # p2_stream = keras.layers.Dense(32, name='p2_dense')(p2_stream)
    
    # ball_stream = keras.layers.LSTM(32, return_sequences=False, name='ball_lstm_1')(ball_input)
    # # ball_stream = keras.layers.LSTM(32, name='ball_lstm_2')(ball_stream)
    
    # # Combined interactions.
    # combined = keras.layers.Concatenate(name='combined_features')([p1_stream, p2_stream, ball_stream])
    # interaction_logic = keras.layers.Dense(96, activation='relu', name='interaction_layer_1')(combined)
    # interaction_logic = keras.layers.Dense(64, activation='relu', name='interaction_layer_2')(interaction_logic)
    
    # # Output heads.
    # paddle1_pos_output = keras.layers.Dense(1, activation='linear', name='paddle1_output_1')(p1_stream)
    # paddle2_pos_output = keras.layers.Dense(1, activation='linear', name='paddle2_output_1')(p2_stream)
    # ball_state_output = keras.layers.Dense(2, activation='linear', name='ball_output_2')(interaction_logic)
    # game_state_output = keras.layers.Dense(1, activation='sigmoid', name='game_state_output_2')(interaction_logic)

    # model = keras.Model(
    #     inputs=[p1_input, p2_input, ball_input],
    #     outputs=[paddle1_pos_output, paddle2_pos_output, ball_state_output, game_state_output],
    # )



    main_input = keras.Input(shape=(24,), name='main_input')

    normalization_layer = keras.layers.Normalization(axis=-1, name='input_normalization')
    normalized_input = normalization_layer(main_input)

    # Branch for the paddle1 features.
    paddle1_features = keras.ops.take(normalized_input, indices=[0, 4, 6, 10, 12, 16, 18, 22], axis=1)
    paddle1_branch = keras.layers.Dense(32, activation='relu', name='paddle1_1')(paddle1_features)
    
    # Branch for the paddle2 features.
    paddle2_features = keras.ops.take(normalized_input, indices=[1, 5, 7, 11, 13, 17, 19, 23], axis=1)
    paddle2_branch = keras.layers.Dense(32, activation='relu', name='paddle2_1')(paddle2_features)
    
    # Branch for ball features.
    ball_features = keras.ops.take(normalized_input, indices=[2, 3, 8, 9, 14, 15, 20, 21], axis=1)
    ball_branch = keras.layers.Dense(64, activation='relu', name='ball_1')(ball_features)
    ball_branch = keras.layers.Dense(64, activation='relu', name='ball_2')(ball_branch)
    ball_branch = keras.layers.Dense(64, activation='relu', name='ball_3')(ball_branch)
    
    # Combine ball with paddle features.
    combined_features = keras.layers.Concatenate(name='concatenate_branches')([ball_branch, paddle1_branch, paddle2_branch])
    shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_1')(combined_features)
    shared_branch = keras.layers.Dense(64, activation='relu', name='ball_paddle_2')(shared_branch)

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
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    
    model.compile(optimizer=optimizer, 
        loss={
            'paddle1_output_1': 'mse',
            'paddle2_output_1': 'mse',
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
