import numpy as np
import random
from random import randrange


def generate_pong_data(num_games=5000, max_frames_per_game=2000, output_filename="data_sim/pong_training_data.txt"):
    """
    Generates synthetic training data for a Pong model by simulating a game.
    
    Args:
        num_games (int): The number of games to simulate.
        max_frames_per_game (int): The maximum number of frames for a single game.
        output_filename (str): The name of the file to save the data to.
    """
    
    # Game constants based on user-defined rules.
    PADDLE_Y_MIN, PADDLE_Y_MAX = 60, 540
    BALL_X_MIN, BALL_X_MAX = 19, 781
    BALL_Y_MIN, BALL_Y_MAX = 9, 591
    PADDLE_HEIGHT = 100
    HALF_PAD_HEIGHT = PADDLE_HEIGHT / 2
    
    # Paddle velocities.
    PADDLE_VELOCITY_OPTIONS = [-8, 0, 8]
    
    # Ball velocities.
    BALL_VELOCITY_OPTIONS = [-3, 3]
    
    # Number of frames to save after a game-over event.
    post_game_frames = 5
    
    file_counter = 1

    for game_idx in range(num_games):
        # Create a new filename for each game with a numbered postfix.
        base_name, extension = output_filename.rsplit('.', 1)
        current_output_filename = f"{base_name}_{str(file_counter).zfill(3)}.{extension}"
        
        # --- Game Initialization ---
        paddle1_pos = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)
        paddle2_pos = random.randint(PADDLE_Y_MIN, PADDLE_Y_MAX)
        
        # Start the ball near the center to avoid immediate boundary issues.
        ball_x = 400
        ball_y = 300

        # Randomly adjust the ball's starting position for more data variety.
        ball_x_adjust = random.randrange(0, 100) * 3
        flip = random.choice([-1, 1])
        ball_x += ball_x_adjust * flip
        ball_y_adjust = random.randrange(0, 66) * 3
        flip = random.choice([-1, 1])
        ball_y += ball_y_adjust * flip

        ball_vx = random.choice(BALL_VELOCITY_OPTIONS)
        ball_vy = random.choice(BALL_VELOCITY_OPTIONS)
        
        while ball_vx == 0:
            ball_vx = random.choice(BALL_VELOCITY_OPTIONS)
        while ball_vy == 0:
            ball_vy = random.choice(BALL_VELOCITY_OPTIONS)
            
        # These velocities represent the movement that will occur in the *next* frame.
        next_paddle1_vel = 0
        next_paddle2_vel = 0
        
        game_state = 0 # 0 = in progress, 1 = game over
        frames_since_game_over = 0

        print(f"Starting game {file_counter}. Writing to {current_output_filename}...")
        
        with open(current_output_filename, 'w') as f:
            for frame_idx in range(max_frames_per_game):
                # --- Capture the Current Frame and Format as a String ---
                # This line captures the current state, including the velocities
                # that will be used to move to the next frame.
                line = f"{paddle1_pos} {paddle2_pos} {ball_x} {ball_y} {next_paddle1_vel} {next_paddle2_vel} {game_state}"
                f.write(line + '\n')
                
                # --- Check for Game Over Condition and Freeze State ---
                if game_state == 1:
                    frames_since_game_over += 1
                    if frames_since_game_over >= post_game_frames:
                        break
                    
                # --- Update Positions for the NEXT Frame ---
                # Update paddle positions using the velocities from the *previous* frame's calculation.
                paddle1_pos += next_paddle1_vel
                paddle2_pos += next_paddle2_vel
                
                # Clamp paddle positions to stay within bounds.
                paddle1_pos = np.clip(paddle1_pos, PADDLE_Y_MIN, PADDLE_Y_MAX)
                paddle2_pos = np.clip(paddle2_pos, PADDLE_Y_MIN, PADDLE_Y_MAX)

                # --- Accurate Ball Collision Detection and Movement ---
                if game_state == 0:
                    # Check for top/bottom wall collisions
                    if ball_y + ball_vy <= BALL_Y_MIN:
                        ball_y = BALL_Y_MIN
                        ball_vy *= -1
                    elif ball_y + ball_vy >= BALL_Y_MAX:
                        ball_y = BALL_Y_MAX
                        ball_vy *= -1
                    else:
                        ball_y += ball_vy
        
                    # Check for left/right collisions (including paddles)
                    if ball_x + ball_vx <= BALL_X_MIN:
                        ball_x = BALL_X_MIN
                        if paddle1_pos - HALF_PAD_HEIGHT <= ball_y <= paddle1_pos + HALF_PAD_HEIGHT:
                            ball_vx *= -1
                        else:
                            game_state = 1
                    elif ball_x + ball_vx >= BALL_X_MAX:
                        ball_x = BALL_X_MAX
                        if paddle2_pos - HALF_PAD_HEIGHT <= ball_y <= paddle2_pos + HALF_PAD_HEIGHT:
                            ball_vx *= -1
                        else:
                            game_state = 1
                    else:
                        ball_x += ball_vx
                else:
                    # If game is over, ball position is frozen.
                    pass
                
                # --- Calculate Velocities for the NEXT Frame's Output ---
                # These are the velocities that will be written in the next iteration.
                if game_state == 0:
                    # Paddle AI for paddle 1 (computer controlled).
                    if ball_y > paddle1_pos + HALF_PAD_HEIGHT:
                        next_paddle1_vel = 8
                    elif ball_y < paddle1_pos - HALF_PAD_HEIGHT:
                        next_paddle1_vel = -8
                    else:
                        next_paddle1_vel = 0
                    
                    # Paddle 2 is a simple AI that moves randomly.
                    next_paddle2_vel = random.choice(PADDLE_VELOCITY_OPTIONS)
                else:
                    next_paddle1_vel = 0
                    next_paddle2_vel = 0
        
        file_counter += 1
            
    print("\nData generation complete.")

if __name__ == "__main__":
    generate_pong_data()
