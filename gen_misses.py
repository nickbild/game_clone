import random


BALL_X_MIN, BALL_X_MAX = 19, 781
BALL_Y_MIN, BALL_Y_MAX = 9, 591
PADDLE_Y_MIN, PADDLE_Y_MAX = 60, 540
PADDLE_HALF_HEIGHT = 50


with open("game_data_misses.txt", 'w') as f:
    for _ in range(5000):
        ###
        # Left paddle miss.
        ###

        try:
            # Start with random paddle positions within the allowed range.
            p1 = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)
            p2 = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)

            # Ball y velocity.
            byv = random.choice([-3, 3])

            # Find valid 'miss' y-coordinates for the ball.
            if byv == -3:
                by_above = random.randint(BALL_Y_MIN+15, p1-50)
                by_below = random.randint(p1+50+15, BALL_Y_MAX)
            else:
                by_above = random.randint(BALL_Y_MIN, p1-50-15)
                by_below = random.randint(p1+50, BALL_Y_MAX-15)

            if random.choice([0, 1]):
                by = by_above
            else:
                by = by_below

            # Generate 5 game frames.
            for i in range(5):
                # Ball x-coords are fixed in this scenario.
                bx = 31 - (3 * i)

                # Pick a random paddle velocity for each frame.
                p1v = random.choice([-8, 0, 8])
                p2v = random.choice([-8, 0, 8])
                
                # Set the game state. 0 until the last frame.
                gs = 0
                if i == 4:
                    gs = 1

                # Add paddle velocities from previous frame to determine current position.
                if i > 0:
                    p1 += p1v_last
                    p2 += p2v_last

                f.write("{0} {1} {2} {3} {4} {5} {6}\n".format(p1, p2, bx, by, p1v, p2v, gs))

                # Save the paddle velocities from the previous frame.
                p1v_last = p1v
                p2v_last = p2v

                # Update ball y-position.
                by += byv
        except:
            # Ignore impossible coordinates.
            print("Out of range.")
            continue

        ###
        # Right paddle miss.
        ###

        try:
            # Start with random paddle positions within the allowed range.
            p1 = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)
            p2 = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)

            # Ball y velocity.
            byv = random.choice([-3, 3])

            # Find valid 'miss' y-coordinates for the ball.
            if byv == -3:
                by_above = random.randint(BALL_Y_MIN+15, p2-50)
                by_below = random.randint(p2+50+15, BALL_Y_MAX)
            else:
                by_above = random.randint(BALL_Y_MIN, p2-50-15)
                by_below = random.randint(p2+50, BALL_Y_MAX-15)

            if random.choice([0, 1]):
                by = by_above
            else:
                by = by_below

            # Generate 5 game frames.
            for i in range(5):
                # Ball x-coords are fixed in this scenario.
                bx = 769 + (3 * i)

                # Pick a random paddle velocity for each frame.
                p1v = random.choice([-8, 0, 8])
                p2v = random.choice([-8, 0, 8])
                
                # Set the game state. 0 until the last frame.
                gs = 0
                if i == 4:
                    gs = 1

                # Add paddle velocities from previous frame to determine current position.
                if i > 0:
                    p1 += p1v_last
                    p2 += p2v_last

                f.write("{0} {1} {2} {3} {4} {5} {6}\n".format(p1, p2, bx, by, p1v, p2v, gs))

                # Save the paddle velocities from the previous frame.
                p1v_last = p1v
                p2v_last = p2v

                # Update ball y-position.
                by += byv
        except:
            # Ignore impossible coordinates.
            print("Out of range.")
            continue

#     f.write("""148 332 31 225 0 0 0
# 148 332 28 222 0 0 0
# 148 332 25 219 0 0 0
# 148 332 22 216 0 0 0
# 148 332 19 213 0 0 1
# """)

#     # Right paddle miss.
#     f.write("""460 476 769 543 0 0 0
# 460 476 772 540 0 0 0
# 460 476 775 537 0 0 0
# 460 476 778 534 0 0 0
# 460 476 781 531 0 0 1
# """)
