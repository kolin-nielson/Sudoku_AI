import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape
from collections import deque
import random
import math
import time

# Hyperparameters
EPISODES = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 2000
BATCH_SIZE = 2048
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 100

# Sudoku grid size
GRID_SIZE = 9

# CNN-based model
def build_model():
    model = Sequential([
        Input(shape=(GRID_SIZE, GRID_SIZE, 1)),  # 9x9 grid with 1 channel
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(GRID_SIZE * GRID_SIZE * GRID_SIZE, activation='softmax'),  # 9 options for each of 81 cells
        Reshape((GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Output shape: (9, 9, 9)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Sudoku environment
class SudokuEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.reset()

    def reset(self):
        # Generate a valid Sudoku puzzle (20% filled for easy difficulty)
        self.grid = generate_puzzle(difficulty=0.2)
        return self.grid.copy()

    def step(self, action):
        row, col, num = action
        if self.is_valid_move(row, col, num):
            self.grid[row, col] = num
            reward = self.calculate_reward()
            done = self.is_solved()
            return self.grid.copy(), reward, done
        else:
            return self.grid.copy(), -0.1, False  # Penalize invalid moves

    def is_valid_move(self, row, col, num):
        # Check row, column, and 3x3 box
        return (num not in self.grid[row, :] and
                num not in self.grid[:, col] and
                num not in self.grid[(row//3)*3:(row//3)*3+3, (col//3)*3:(col//3)*3+3])

    def calculate_reward(self):
        # Reward for correct digits and penalize conflicts
        reward = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] != 0:
                    if self.is_valid_move(i, j, self.grid[i, j]):
                        reward += 0.1  # Small reward for correct digits
                    else:
                        reward -= 0.1  # Penalty for conflicts
        return reward

    def is_solved(self):
        # Check if the puzzle is solved
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == 0 or not self.is_valid_move(i, j, self.grid[i, j]):
                    return False
        return True

# Generate a valid Sudoku puzzle
def generate_puzzle(difficulty=0.2):
    # Generate a solved Sudoku grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    solve_sudoku(grid)
    # Mask cells to create a puzzle
    mask = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[difficulty, 1 - difficulty])
    return grid * mask

# Sudoku solver (backtracking)
def solve_sudoku(grid):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == 0:
                for num in range(1, GRID_SIZE + 1):
                    if is_valid(grid, i, j, num):
                        grid[i, j] = num
                        if solve_sudoku(grid):
                            return True
                        grid[i, j] = 0
                return False
    return True

# Check if a move is valid
def is_valid(grid, row, col, num):
    for i in range(GRID_SIZE):
        if grid[row, i] == num or grid[i, col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if grid[start_row + i, start_col + j] == num:
                return False
    return True

# Main training loop
def train():
    env = SudokuEnv()
    model = build_model()
    target_model = build_model()
    target_model.set_weights(model.get_weights())
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    for episode in range(EPISODES):
        state = env.reset()
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = (np.random.randint(GRID_SIZE), np.random.randint(GRID_SIZE), np.random.randint(1, GRID_SIZE + 1))
            else:
                q_values = model.predict(state[np.newaxis, ...])
                action = np.unravel_index(np.argmax(q_values), (GRID_SIZE, GRID_SIZE, GRID_SIZE))

            next_state, reward, done = env.step(action)
            total_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Train on a batch from the replay buffer
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                next_states = np.array(next_states)

                # Predict Q-values for next states
                future_q_values = target_model.predict(next_states)
                target_q_values = rewards + (1 - np.array(dones)) * GAMMA * np.max(future_q_values, axis=(1, 2, 3))

                # Update Q-values for current states
                q_values = model.predict(states)
                for i, (row, col, num) in enumerate(actions):
                    q_values[i, row, col, num - 1] = target_q_values[i]

                # Train the model
                model.train_on_batch(states, q_values)

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())

        # Log progress
        print(f"Episode {episode} | Epsilon: {epsilon:.3f} | Avg Reward: {total_reward:.2f}")

# Run training
if __name__ == "__main__":
    train()