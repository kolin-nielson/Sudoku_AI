# Sudoku Reinforcement Learning Visualizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.x-brightgreen)
![Gym](https://img.shields.io/badge/Gym-0.21-blueviolet)

## Overview

**Sudoku Reinforcement Learning Visualizer** is an experimental project that demonstrates how an AI system can learn to solve Sudoku puzzles using reinforcement learning (RL). Unlike traditional supervised approaches, our system trains an RL agent with Deep Q‑Learning (DQN) that interacts with a custom OpenAI Gym environment. The training process and the final puzzle-solving results are displayed visually via a Pygame interface, so you can watch the agent learn and evolve over time.

## What Is This AI System?

This AI system is built around a reinforcement learning agent that learns through trial and error:
- **Agent:** Uses Deep Q‑Learning (DQN) to choose actions that fill in cells on a Sudoku board.
- **Environment:** A custom Gym environment models a Sudoku puzzle. The state is the current board (as a flattened 81‑element vector), and the action space consists of 729 discrete actions—each corresponding to filling a specific cell with a digit (1–9).
- **Reward System:**  
  - A small negative reward for each move encourages the agent to solve the puzzle in as few moves as possible.  
  - A penalty for invalid moves (or trying to change fixed clues) teaches the agent to respect Sudoku’s rules.  
  - A large positive reward is given for completely and correctly solving the puzzle.
- **Experience Replay & Target Network:**  
  To stabilize training, the agent uses experience replay and a periodically updated target network.

## How Does It Work?

1. **Environment Setup:**  
   The custom Gym environment generates a valid Sudoku puzzle by first creating a complete board and then removing a subset of numbers. This serves as the state for the RL agent.

2. **Deep Q‑Learning Agent:**  
   - **State:** The current Sudoku board (flattened into 81 numbers normalized between 0 and 1).  
   - **Action:** The agent selects one out of 729 actions (9 cells × 9 possible digits).  
   - **Q‑Values:** The DQN estimates Q‑values for every possible action. During training, the agent uses an epsilon‑greedy strategy to balance exploration and exploitation.
   - **Learning:**  
     The agent samples past experiences from a replay buffer and learns by minimizing the difference between predicted Q‑values and target Q‑values (using the Bellman equation).

3. **Visualization:**  
   - **Live Training:**  
     The training loop is visualized using Pygame. At each episode, you see the current board state, the agent’s actions, and live training statistics (episode number, total reward, epsilon, estimated remaining training time).
   - **Demo Mode:**  
     After training, you can run the agent on new puzzles and watch as it fills in the board, all rendered in a modern, clean interface.

## Why Does It Work?

- **Learning Through Interaction:**  
  Unlike supervised learning that requires labeled examples (puzzle–solution pairs), our RL agent learns by interacting with the environment. It receives feedback (rewards or penalties) based on its actions, enabling it to explore different strategies and gradually learn the complex rules of Sudoku.

- **Trial-and-Error:**  
  The reward system is designed so that the agent is incentivized to find a solution as quickly as possible while avoiding invalid moves. Over many episodes, the agent learns which actions lead to higher rewards and which do not.

- **Stabilization Techniques:**  
  - **Experience Replay:**  
    Reusing past experiences helps to break correlations between sequential moves and leads to more stable and efficient learning.
  - **Target Network:**  
    Periodically updating a separate target network reduces oscillations and divergence during training.

- **Visualization Enhances Understanding:**  
  By seeing the agent’s progress in real time, you can observe how it explores the solution space, adjusts its strategy, and eventually converges to solving the puzzle. This visual component makes the training process more transparent and engaging.

## Technologies Used

- **Python:** The core programming language.
- **TensorFlow & Keras:** To build and train the DQN model.
- **OpenAI Gym:** For creating the custom Sudoku environment.
- **Pygame:** For a modern and interactive visualization of both training and demonstration.
- **NumPy:** For efficient numerical operations.
