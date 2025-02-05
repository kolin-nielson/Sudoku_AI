# train_visual.py
import os
import time
import random
import numpy as np
import pygame
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sudoku_env import SudokuEnv

# Enable mixed precision for GPU optimization
from tensorflow.keras.mixed_precision import set_global_policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
set_global_policy(policy)

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Hyperparameters
EPISODES = 5000
MAX_STEPS = 500
BATCH_SIZE = 256  # Increased for GPU
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 10
MEMORY_SIZE = 100000
NUM_ENVS = 4

# Pygame settings
WIN_WIDTH, WIN_HEIGHT = 800, 900
BOARD_AREA_SIZE = 800
INFO_HEIGHT = WIN_HEIGHT - BOARD_AREA_SIZE
GRID_ROWS, GRID_COLS = 2, 2
MARGIN = 10
SUB_BOARD_WIDTH = (BOARD_AREA_SIZE - (GRID_COLS + 1) * MARGIN) // GRID_COLS
SUB_BOARD_HEIGHT = (BOARD_AREA_SIZE - (GRID_ROWS + 1) * MARGIN) // GRID_ROWS

# Colors
BG_COLOR = (248, 248, 248)
GRID_COLOR = (50, 50, 50)
CELL_BG_COLOR = (255, 255, 255)
TEXT_COLOR = (44, 62, 80)
INFO_BG_COLOR = (230, 230, 230)
INFO_BORDER_COLOR = (50, 50, 50)
SUB_BOARD_BORDER_COLOR = (0, 0, 0)

# Build DQN model
def build_dqn_model():
    model = Sequential()
    model.add(Dense(256, input_dim=81, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(9 * 9 * 9, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

# Update Pygame screen
def update_screen(screen, board_matrices, info_text, dialog_text, board_font, info_font, dialog_font):
    """
    Update the entire Pygame window:
      - The top BOARD_AREA_SIZE x BOARD_AREA_SIZE area is divided into a 2x2 grid.
      - Each sub-board is drawn with margins, a thick border, and a label.
      - The bottom INFO_HEIGHT area displays training info and a dialog message.
    """
    # Clear the screen
    screen.fill(BG_COLOR)

    # Draw each sub-board
    for idx, board in enumerate(board_matrices):
        # Determine grid position
        row_idx = idx // GRID_COLS
        col_idx = idx % GRID_COLS
        offset_x = MARGIN + col_idx * (SUB_BOARD_WIDTH + MARGIN)
        offset_y = MARGIN + row_idx * (SUB_BOARD_HEIGHT + MARGIN)

        # Draw sub-board background
        sub_rect = pygame.Rect(offset_x, offset_y, SUB_BOARD_WIDTH, SUB_BOARD_HEIGHT)
        pygame.draw.rect(screen, CELL_BG_COLOR, sub_rect)

        # Draw each cell in the sub-board
        cell_w = SUB_BOARD_WIDTH / 9
        cell_h = SUB_BOARD_HEIGHT / 9
        for i in range(9):
            for j in range(9):
                cell_rect = pygame.Rect(offset_x + j * cell_w, offset_y + i * cell_h, cell_w, cell_h)
                pygame.draw.rect(screen, CELL_BG_COLOR, cell_rect)
                pygame.draw.rect(screen, GRID_COLOR, cell_rect, 1)
                num = board[i][j]
                if num != 0:
                    text = board_font.render(str(num), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=(offset_x + j * cell_w + cell_w/2,
                                                      offset_y + i * cell_h + cell_h/2))
                    screen.blit(text, text_rect)

        # Draw a thick border around the sub-board
        pygame.draw.rect(screen, SUB_BOARD_BORDER_COLOR, sub_rect, 4)
        # Label the sub-board
        label = f"Env {idx+1}"
        label_surface = info_font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (offset_x + 5, offset_y + 5))

    # Draw the Info Area
    info_rect = pygame.Rect(0, BOARD_AREA_SIZE, WIN_WIDTH, INFO_HEIGHT)
    pygame.draw.rect(screen, INFO_BG_COLOR, info_rect)
    pygame.draw.rect(screen, INFO_BORDER_COLOR, info_rect, 2)
    info_surface = info_font.render(info_text, True, TEXT_COLOR)
    screen.blit(info_surface, (10, BOARD_AREA_SIZE + 10))

    # Draw the Dialog Area
    dialog_rect = pygame.Rect(10, BOARD_AREA_SIZE + 40, WIN_WIDTH - 20, INFO_HEIGHT - 50)
    pygame.draw.rect(screen, BG_COLOR, dialog_rect)
    pygame.draw.rect(screen, INFO_BORDER_COLOR, dialog_rect, 2)
    dialog_surface = dialog_font.render(dialog_text, True, TEXT_COLOR)
    dialog_text_rect = dialog_surface.get_rect(center=dialog_rect.center)
    screen.blit(dialog_surface, dialog_text_rect)

    pygame.display.flip()

# Training loop
def main():
    envs = [SudokuEnv(removals=40) for _ in range(NUM_ENVS)]
    states = [env.reset() / 9.0 for env in envs]
    dones = [False] * NUM_ENVS
    total_rewards = [0] * NUM_ENVS
    memory = deque(maxlen=MEMORY_SIZE)
    model = build_dqn_model()
    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())
    epsilon = EPSILON_START
    start_time = time.time()

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Sudoku RL Training Visualization")
    board_font = pygame.font.SysFont("Helvetica", 28)
    info_font = pygame.font.SysFont("Helvetica", 20)
    dialog_font = pygame.font.SysFont("Helvetica", 18)

    for e in range(EPISODES):
        for i in range(NUM_ENVS):
            if dones[i]:
                states[i] = envs[i].reset() / 9.0
                total_rewards[i] = 0
                dones[i] = False

        step = 0
        while step < MAX_STEPS and not all(dones):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            for i in range(NUM_ENVS):
                if dones[i]:
                    continue
                if np.random.rand() <= epsilon:
                    action = random.randrange(envs[i].action_space.n)
                else:
                    q_values = model.predict(states[i].reshape(1, -1), verbose=0)
                    action = np.argmax(q_values[0])
                
                next_state, reward, done, _ = envs[i].step(action)
                next_state = next_state / 9.0
                memory.append((states[i], action, reward, next_state, done))
                states[i] = next_state
                total_rewards[i] += reward
                dones[i] = done

            step += 1

            # Update visualization
            board_matrices = [np.array(state * 9.0, dtype=np.int32).reshape(9, 9) for state in states]
            avg_reward = sum(total_rewards) / NUM_ENVS
            est_remaining = (time.time() - start_time) / (e + 1) * (EPISODES - e - 1)
            info_text = (f"Episode: {e}  Step: {step}  Avg Reward: {avg_reward:.2f}  "
                         f"Epsilon: {epsilon:.3f}  Est. Time Remain: {est_remaining/60:.2f} min")
            dialog_text = ""
            update_screen(screen, board_matrices, info_text, dialog_text, board_font, info_font, dialog_font)
            pygame.time.delay(100)

        # Experience replay
        if len(memory) >= BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            states_mb = np.array([m[0] for m in minibatch])
            actions_mb = np.array([m[1] for m in minibatch])
            rewards_mb = np.array([m[2] for m in minibatch])
            next_states_mb = np.array([m[3] for m in minibatch])
            dones_mb = np.array([m[4] for m in minibatch])
            target = model.predict(states_mb, verbose=0)
            target_next = target_model.predict(next_states_mb, verbose=0)
            for j in range(BATCH_SIZE):
                if dones_mb[j]:
                    target[j][actions_mb[j]] = rewards_mb[j]
                else:
                    target[j][actions_mb[j]] = rewards_mb[j] + GAMMA * np.amax(target_next[j])
            model.fit(states_mb, target, epochs=1, verbose=0)

        # Update target network
        if e % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())

        # Decay epsilon
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # Print progress
        if e % 100 == 0:
            print(f"Episode {e}/{EPISODES} - Avg Reward: {avg_reward:.2f} - Epsilon: {epsilon:.3f} - Estimated Time Remaining: {est_remaining/60:.2f} min")

    model.save("sudoku_dqn_model.h5")
    print("Model saved as sudoku_dqn_model.h5")
    pygame.quit()

if __name__ == "__main__":
    main()