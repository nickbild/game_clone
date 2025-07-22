from tensorflow import keras
import numpy as np
import pygame
from pygame.locals import *
import sys


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

# fps = pygame.time.Clock()
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

# Set the initial game state.
past_positions = [[300, 300, 403, 297, 0, 0], 
                  [300, 300, 406, 294, 0, 0], 
                  [300, 300, 409, 291, 0, 0], 
                  [300, 300, 412, 288, 0, 0], 
                  [300, 300, 415, 285, 0, 0]]
past_positions = np.asarray(past_positions).astype('float32').reshape(1, 5, 6)

# Load the saved model.
model = keras.models.load_model("pong.keras")

while True:
    # Predict the next frame.
    new_prediction = model.predict(past_positions, verbose=0)

    # Capture state of controls.
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Rotate the latest prediction into the past positions array.
    past_positions[0][0] = past_positions[0][1]
    past_positions[0][1] = past_positions[0][2]
    past_positions[0][2] = past_positions[0][3]
    past_positions[0][3] = past_positions[0][4]
    past_positions[0][4] = np.append(np.rint(new_prediction[0]), [paddle1_vel, paddle2_vel])

    # Update the display.
    window.fill(BLACK)
    pygame.draw.circle(window, RED, [int(past_positions[0][4][2]), int(past_positions[0][4][3])], BALL_RADIUS, 0)
    pygame.draw.polygon(window, RED, [[0, int(past_positions[0][4][0]) - HALF_PAD_HEIGHT], [0, int(past_positions[0][4][0]) + HALF_PAD_HEIGHT], [9, int(past_positions[0][4][0]) + HALF_PAD_HEIGHT], [9, int(past_positions[0][4][0]) - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(window, RED, [[790, int(past_positions[0][4][1]) - HALF_PAD_HEIGHT], [790, int(past_positions[0][4][1]) + HALF_PAD_HEIGHT], [799, int(past_positions[0][4][1]) + HALF_PAD_HEIGHT], [799, int(past_positions[0][4][1]) - HALF_PAD_HEIGHT]], 0)
    pygame.display.update()

    print(past_positions)

    # fps.tick(60)
