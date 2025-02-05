# Sudoku Reinforcement Learning Visualizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.x-brightgreen)
![Gym](https://img.shields.io/badge/Gym-0.21-blueviolet)

## Overview

**Sudoku Reinforcement Learning Visualizer** is an experimental project that demonstrates how an AI system can learn to solve Sudoku puzzles using reinforcement learning (RL). The project is designed not only to build a powerful learning agent but also to make the training process engaging and accessible through a visual interface. In this project, you’ll see a live display of multiple parallel Sudoku environments as the RL agent learns from its actions.

## What Is the AI System?

The AI system is based on a reinforcement learning agent that uses Deep Q‑Learning (DQN) to solve Sudoku puzzles. Here's a simple breakdown:

- **Agent:**  
  The agent is a neural network (a deep Q-network) that looks at the current state of the Sudoku board and decides which cell to fill with which number.  
- **Environment:**  
  The environment is a custom version of an OpenAI Gym environment designed for Sudoku. It generates puzzles, simulates moves, and provides feedback in the form of rewards.
- **Action Space:**  
  There are 729 possible actions. Each action corresponds to choosing a specific cell (out of 81) and filling it with a digit (1–9).

## How Reinforcement Learning Works Here

Reinforcement learning is all about learning by trial and error. The agent interacts with the environment in the following way:

1. **Observing the State:**  
   The agent sees the current Sudoku board (with empty cells and some numbers filled in). This board is represented as an 81-element list (flattened from the 9×9 grid).

2. **Selecting an Action:**  
   The agent uses an **epsilon-greedy strategy**:
   - With probability **ε (epsilon)**, it chooses a random move to explore new possibilities.
   - With probability **1 - ε**, it chooses the move that it believes will yield the highest reward (based on its current understanding).
   
   **Epsilon (ε)** starts high (encouraging lots of exploration) and gradually decays over time, so the agent eventually relies on its learned knowledge.

3. **Receiving a Reward:**  
   After making a move, the environment gives the agent a reward:
   - **Small Negative Reward (-0.1):** A tiny penalty for every move encourages the agent to solve the puzzle in as few moves as possible.
   - **Penalty for Invalid Moves (-1):** If the agent tries to change a fixed clue or make a move that violates Sudoku rules (e.g., placing a duplicate number in a row, column, or 3×3 block), it receives a -1 penalty.
   - **Big Positive Reward (+100):** If the agent fills in all cells correctly (a valid solution), it gets a large reward of +100.
   - **Moderate Penalty (-10):** If the board is completely filled but the solution is invalid, the agent gets a penalty of -10.
   
   This reward structure guides the agent to learn the rules of Sudoku and to prefer moves that lead to a valid solution quickly.

4. **Learning from Experience:**  
   The agent stores its experiences (state, action, reward, next state, done) in a replay buffer. It then samples batches from this buffer to train the neural network, helping it to learn which moves lead to higher rewards.

5. **Target Network:**  
   A separate target network is used to stabilize training. This network is updated periodically with the weights of the main network.

## The Reward System – Why It’s Set Up This Way

The reward system is critical because it tells the agent which behaviors are good and which are bad:
- **Efficiency:**  
  The small per-move penalty (-0.1) encourages the agent to solve the puzzle with fewer moves.
- **Rule Adherence:**  
  Heavier penalties for invalid moves (-1) ensure the agent learns that certain moves (like altering fixed clues or violating the uniqueness rule) are unacceptable.
- **Success Signal:**  
  A large reward (+100) for solving the puzzle correctly provides a strong incentive for the agent to find the correct sequence of moves.
- **Failure Feedback:**  
  A moderate penalty (-10) for a filled but invalid board signals that simply filling all the cells is not enough—correctness is crucial.

## Visual and Immersive Training

Our project doesn’t just train an agent—it shows you the training process in real time:
- **Multiple Environments:**  
  The top portion of the window displays four parallel Sudoku boards (arranged in a 2×2 grid). Each board represents a separate environment where the agent is interacting with a puzzle.
- **Clear Layout:**  
  Each sub-board is separated by margins, has a thick border, and is clearly labeled (e.g., "Env 1", "Env 2", etc.) so you can easily see where one environment ends and the next begins.
- **Training Statistics:**  
  The bottom portion of the window displays key training information such as the current episode number, step count, average reward across environments, current epsilon value, and an estimated remaining training time.
- **Professional Appearance:**  
  With a larger window, clear fonts, well-organized layout, and neat borders, the visualization is designed to be both informative and appealing—even to those new to AI.

## Why It Works

- **Trial and Error:**  
  The agent learns by trying out different moves and receiving feedback via rewards. Over time, it learns which actions lead to a solved puzzle.
- **Balancing Exploration and Exploitation:**  
  The epsilon-greedy strategy ensures that the agent explores various moves early on and gradually shifts to exploiting the best moves as it learns.
- **Stable Learning:**  
  Techniques like experience replay and the use of a target network help to stabilize the learning process.
- **Immersive Visualization:**  
  Watching multiple environments simultaneously gives a clear picture of how the agent learns from its actions. The training stats and layout make it accessible and engaging.

## Technologies Used

- **Python:** The programming language powering the entire project.
- **TensorFlow & Keras:** For building and training the deep Q-network.
- **OpenAI Gym:** For creating the custom environment where the agent learns to solve Sudoku.
- **Pygame:** For visualizing the training process in a professional and engaging way.
- **NumPy:** For efficient numerical operations and data handling.
