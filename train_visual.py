# train_visual.py
import pygame
import numpy as np
import random
import time
from collections import deque
import gym
from sudoku_env import SudokuEnv
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Hyperparameters
EPISODES = 5000            # Total training episodes (increase as needed)
MAX_STEPS = 500            # Maximum steps per episode
BATCH_SIZE = 64
GAMMA = 0.99               # Discount factor for future rewards
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 10    # Update target network every N episodes
MEMORY_SIZE = 100000

# Pygame window settings (for visualization)
WIDTH, HEIGHT = 540, 750
GRID_SIZE = 540
CELL_SIZE = GRID_SIZE // 9
BG_COLOR = (248, 248, 248)
GRID_COLOR = (50, 50, 50)
CELL_BG_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (241, 196, 15)
TEXT_COLOR = (44, 62, 80)

def build_dqn_model():
    # Build a simple deep Q-network.
    model = Sequential()
    model.add(Dense(256, input_dim=81, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(9 * 9 * 9, activation='linear'))  # 729 actions
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model

def estimate_time(start_time, current_episode, total_episodes):
    elapsed = time.time() - start_time
    remaining = (elapsed / (current_episode + 1)) * (total_episodes - current_episode - 1)
    return remaining

def draw_board(screen, board, font):
    # Draw the Sudoku board on the Pygame window.
    screen.fill(BG_COLOR)
    for i in range(9):
        for j in range(9):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, CELL_BG_COLOR, rect)
            # Draw grid lines
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
            # Draw number if present
            num = board[i][j]
            if num != 0:
                text = font.render(str(num), True, TEXT_COLOR)
                text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, text_rect)
    pygame.display.update()

def draw_info(screen, episode, total_reward, epsilon, est_remaining, font, small_font):
    # Draw training info (episode, total reward, epsilon, estimated remaining time)
    info = f"Episode: {episode}  Reward: {total_reward:.2f}  Epsilon: {epsilon:.3f}"
    time_info = f"Est. Time Remain: {est_remaining/60:.2f} min"
    text = font.render(info, True, TEXT_COLOR)
    time_text = small_font.render(time_info, True, TEXT_COLOR)
    screen.blit(text, (10, GRID_SIZE + 10))
    screen.blit(time_text, (10, GRID_SIZE + 50))
    pygame.display.update()

def main():
    # Initialize Gym environment and Pygame
    env = SudokuEnv(removals=40)
    memory = deque(maxlen=MEMORY_SIZE)
    model = build_dqn_model()
    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())
    epsilon = EPSILON_START
    start_time = time.time()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sudoku RL Training Visualization")
    font = pygame.font.SysFont("Helvetica", 24)
    small_font = pygame.font.SysFont("Helvetica", 18)
    
    for e in range(EPISODES):
        state = env.reset()
        state = state / 9.0  # Normalize state values to [0, 1]
        total_reward = 0
        done = False
        
        # Visualize the initial board.
        board_matrix = np.array(state * 9.0, dtype=np.int32).reshape(9, 9)
        draw_board(screen, board_matrix, font)
        
        for step in range(MAX_STEPS):
            # Allow early exit from visualization.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Choose an action using epsilon-greedy strategy.
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space.n)
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
                
            next_state, reward, done, _ = env.step(action)
            next_state = next_state / 9.0  # Normalize next state
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Visualize current board state every few steps.
            board_matrix = np.array(state * 9.0, dtype=np.int32).reshape(9, 9)
            draw_board(screen, board_matrix, font)
            pygame.time.delay(50)  # Slow down for visualization
            
            if done:
                break
        
        # Experience replay.
        if len(memory) >= BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            states = np.array([m[0] for m in minibatch])
            actions = np.array([m[1] for m in minibatch])
            rewards = np.array([m[2] for m in minibatch])
            next_states = np.array([m[3] for m in minibatch])
            dones = np.array([m[4] for m in minibatch])
            target = model.predict(states, verbose=0)
            target_next = target_model.predict(next_states, verbose=0)
            for i in range(BATCH_SIZE):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + GAMMA * np.amax(target_next[i])
            model.fit(states, target, epochs=1, verbose=0)
        
        # Update the target network periodically.
        if e % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        
        # Decay epsilon.
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
        
        # Estimate remaining training time.
        est_remaining = estimate_time(start_time, e, EPISODES)
        draw_info(screen, e, total_reward, epsilon, est_remaining, font, small_font)
        
        # Print info to console every 100 episodes.
        if e % 100 == 0:
            print(f"Episode {e}/{EPISODES} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f} - Estimated Time Remaining: {est_remaining/60:.2f} min")
    
    # Save the trained model.
    model.save("sudoku_dqn_model.h5")
    print("Model saved as sudoku_dqn_model.h5")
    pygame.quit()

if __name__ == "__main__":
    main()
