# rl_train.py
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
EPISODES = 10000
MAX_STEPS = 500        # Maximum steps per episode
BATCH_SIZE = 64
GAMMA = 0.99           # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
MEMORY_SIZE = 100000

def build_dqn_model():
    """
    Build a simple deep Q-network.
    Input: flattened board (81 values).
    Output: Q-values for each of the 729 possible actions.
    """
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

def main():
    env = SudokuEnv(removals=40)
    memory = deque(maxlen=MEMORY_SIZE)
    model = build_dqn_model()
    target_model = clone_model(model)
    target_model.set_weights(model.get_weights())
    
    epsilon = EPSILON_START
    start_time = time.time()
    
    for e in range(EPISODES):
        state = env.reset()
        state = state / 9.0  # Normalize state values to [0, 1]
        total_reward = 0
        for step in range(MAX_STEPS):
            # Epsilon-greedy action selection.
            if np.random.rand() <= epsilon:
                action = random.randrange(env.action_space.n)
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state / 9.0  # Normalize next state.
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
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
        
        # Update target network periodically.
        if e % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        
        # Decay epsilon.
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
        
        if e % 100 == 0:
            remaining = estimate_time(start_time, e, EPISODES)
            print(f"Episode {e}/{EPISODES} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f} - Estimated Time Remaining: {remaining/60:.2f} min")
    
    # Save the trained model.
    model.save("sudoku_dqn_model.h5")
    print("Model saved as sudoku_dqn_model.h5")

if __name__ == "__main__":
    main()
