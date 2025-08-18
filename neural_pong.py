import tensorflow as tf
from tensorflow import keras
import numpy as np
import pygame
from pygame.locals import *
import sys


# Constants for calculating new features. These must match the training script.
BALL_X_MIN, BALL_X_MAX = 19, 781
BALL_Y_MIN, BALL_Y_MAX = 9, 591
PADDLE_Y_MIN, PADDLE_Y_MAX = 60, 540
PADDLE_HALF_HEIGHT = 50

# Calculate ranges for normalization
x_range = float(BALL_X_MAX - BALL_X_MIN)
y_range = float(BALL_Y_MAX - BALL_Y_MIN)


def keydown(event):
    global paddle1_vel, paddle2_vel
    
    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_w:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel
    
    if event.key in (K_w, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0


pygame.init()

fps = pygame.time.Clock()
RED = (255,0,0)
BLACK = (0,0,0)
WIDTH = 800
HEIGHT = 600        
BALL_RADIUS = 10
PAD_WIDTH = 10
PAD_HEIGHT = 100
HALF_PAD_WIDTH = PAD_WIDTH / 2
HALF_PAD_HEIGHT = PAD_HEIGHT / 2

paddle1_vel = 0
paddle2_vel = 0

window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('NeuralPong')

# Initialize the past positions with unscaled data.
# This array stores 4 frames of [p1_pos, p2_pos, ball_x, ball_y, p1_vel, p2_vel]
raw_past_positions = np.array([
    [300, 300, 403, 297, 0, 0],
    [300, 300, 406, 294, 0, 0],
    [300, 300, 409, 291, 0, 0],
    [300, 300, 412, 288, 0, 0]
], dtype=np.float32)

# Load the saved model.
# NOTE: The model uses a Lambda layer with a custom function.
# This requires `safe_mode=False` for deserialization.
model = keras.models.load_model("pong.keras", safe_mode=False)


def game_reset():
    global raw_past_positions
    # Re-initialize the raw_past_positions array to a starting state
    raw_past_positions = np.array([
        [300, 300, 403, 297, 0, 0],
        [300, 300, 406, 294, 0, 0],
        [300, 300, 409, 291, 0, 0],
        [300, 300, 412, 288, 0, 0]
    ], dtype=np.float32)
    print("Game Over. Resetting...")


while True:
    # --- Step 1: Process the raw_past_positions to create the input vector ---
    
    # Initialize list for the new features.
    new_features = []

    # Iterate through the first 3 frames to compute deltas
    for i in range(3):
        # ball deltas (velocities)
        delta_x = raw_past_positions[i+1][2] - raw_past_positions[i][2]
        delta_y = raw_past_positions[i+1][3] - raw_past_positions[i][3]
        
        # normalized distances to boundaries and paddle coverages
        ball_x, ball_y = raw_past_positions[i][2], raw_past_positions[i][3]
        p1_pos, p2_pos = raw_past_positions[i][0], raw_past_positions[i][1]

        dist_left = (ball_x - BALL_X_MIN) / x_range
        dist_right = (BALL_X_MAX - ball_x) / x_range
        dist_top = (ball_y - BALL_Y_MIN) / y_range
        dist_bottom = (BALL_Y_MAX - ball_y) / y_range
        coverage_p1 = abs(ball_y - p1_pos) / PADDLE_HALF_HEIGHT
        coverage_p2 = abs(ball_y - p2_pos) / PADDLE_HALF_HEIGHT

        new_features.extend([delta_x, delta_y, dist_left, dist_right, dist_top, dist_bottom, coverage_p1, coverage_p2])

    # Handle the last frame (frame 4) with repeated deltas
    last_delta_x = raw_past_positions[3][2] - raw_past_positions[2][2]
    last_delta_y = raw_past_positions[3][3] - raw_past_positions[2][3]

    ball_x, ball_y = raw_past_positions[3][2], raw_past_positions[3][3]
    p1_pos, p2_pos = raw_past_positions[3][0], raw_past_positions[3][1]

    dist_left = (ball_x - BALL_X_MIN) / x_range
    dist_right = (BALL_X_MAX - ball_x) / x_range
    dist_top = (ball_y - BALL_Y_MIN) / y_range
    dist_bottom = (BALL_Y_MAX - ball_y) / y_range
    coverage_p1 = abs(ball_y - p1_pos) / PADDLE_HALF_HEIGHT
    coverage_p2 = abs(ball_y - p2_pos) / PADDLE_HALF_HEIGHT

    new_features.extend([last_delta_x, last_delta_y, dist_left, dist_right, dist_top, dist_bottom, coverage_p1, coverage_p2])
    
    # Combine original and new features into a single input vector
    full_input_vector = np.concatenate([raw_past_positions.flatten(), new_features], dtype=np.float32).reshape(1, 56)

    # --- Step 2: Make the prediction ---
    new_prediction = model.predict(full_input_vector, verbose=0)
    
    predicted_p1_pos = new_prediction[0][0]
    predicted_p2_pos = new_prediction[1][0]
    predicted_ball_state = new_prediction[2][0]
    game_over_probability = new_prediction[3][0][0]

    # Check for game over state and reset if needed
    if game_over_probability > 0.5: # Use a threshold
         game_reset()
        
    # --- Step 3: Create the new frame using predicted and live data ---
    new_p1_pos = predicted_p1_pos[0]
    new_p2_pos = predicted_p2_pos[0]
    new_ball_x = predicted_ball_state[0]
    new_ball_y = predicted_ball_state[1]

    # Build the complete new frame.
    new_frame = np.array([
        new_p1_pos, new_p2_pos, new_ball_x, new_ball_y, paddle1_vel, paddle2_vel
    ], dtype=np.float32)

    # --- Step 4: Update the raw_past_positions array with the new frame ---
    raw_past_positions = np.roll(raw_past_positions, -1, axis=0)
    raw_past_positions[-1] = new_frame
    
    # --- Step 5: Update Pygame display using the denormalized positions ---
    window.fill(BLACK)
    pygame.draw.circle(window, RED, [int(new_ball_x), int(new_ball_y)], BALL_RADIUS, 0)
    pygame.draw.polygon(window, RED, [[0, int(new_p1_pos) - HALF_PAD_HEIGHT], [0, int(new_p1_pos) + HALF_PAD_HEIGHT], [9, int(new_p1_pos) + HALF_PAD_HEIGHT], [9, int(new_p1_pos) - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(window, RED, [[790, int(new_p2_pos) - HALF_PAD_HEIGHT], [790, int(new_p2_pos) + HALF_PAD_HEIGHT], [799, int(new_p2_pos) + HALF_PAD_HEIGHT], [799, int(new_p2_pos) - HALF_PAD_HEIGHT]], 0)
    pygame.display.update()

    # --- Capture user input and other events ---
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()

    # fps.tick(60)
