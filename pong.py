import random
import pygame, sys
from pygame.locals import *


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
ball_pos = [0,0]
ball_vel = [0,0]
paddle1_vel = 0
paddle2_vel = 0

window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Pong')


def ball_init():
    global ball_pos, ball_vel
    ball_pos = [WIDTH/2, HEIGHT/2]
    horz = 3
    vert = 3
            
    ball_vel = [horz,-vert]


def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel
    global score1, score2
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT/2]
    paddle2_pos = [WIDTH +1 - HALF_PAD_WIDTH, HEIGHT/2]

    ball_init()


def point_scored():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel
           
    canvas.fill(BLACK)

    if paddle1_pos[1] > HALF_PAD_HEIGHT + 10 and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT - 10:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT + 10 and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT - 10 and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel
    
    if paddle2_pos[1] > HALF_PAD_HEIGHT + 10 and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT - 10:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT + 10 and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT - 10 and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel

    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    pygame.draw.circle(canvas, RED, ball_pos, BALL_RADIUS, 0)
    pygame.draw.polygon(canvas, RED, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(canvas, RED, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)

    # Top/bottom collision check.
    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    
    # Gutter/paddle collision check.
    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(int(paddle1_pos[1] - HALF_PAD_HEIGHT - 1), int(paddle1_pos[1] + HALF_PAD_HEIGHT + 5), 1):
        ball_vel[0] = -ball_vel[0]
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        point_scored()
        
    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(int(paddle2_pos[1] - HALF_PAD_HEIGHT - 1), int(paddle2_pos[1] + HALF_PAD_HEIGHT + 5), 1):
        ball_vel[0] = -ball_vel[0]
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        point_scored()


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


init()

f = open("game_data.txt", "a")

# Game loop.
while True:
    draw(window)

    for event in pygame.event.get():
        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()
            
    pygame.display.update()

    f.write("{0} {1} {2} {3} {4} {5}\n".format(int(paddle1_pos[1]), int(paddle2_pos[1]), int(ball_pos[0]), int(ball_pos[1]), int(paddle1_vel), int(paddle2_vel)))

    if ball_pos[0] == 403 and ball_pos[1] == 297 and ball_vel[0] == 3 and ball_vel[1] == -3:
        print("STARTING POINT REACHED")

    fps.tick(60)

f.close()
