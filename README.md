# Sudoku AI Visualizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.x-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Overview

**Sudoku AI Visualizer** is a fun and educational project that combines classic Sudoku puzzles with modern AI techniques and a sleek, professional user interface.  
This project demonstrates:
- **Board Generation:** How to generate a complete, valid Sudoku board and remove numbers to create a playable puzzle.
- **AI Training & Inference:** How to train a neural network model to "guess" the solution to a Sudoku puzzle and see its reasoning process.
- **Hybrid Solving:** A unique approach where the AI's predictions are animated and then verified by a traditional backtracking solver to ensure accuracy.
- **Modern UI:** A visually appealing and interactive interface built with Pygame that brings the whole process to life.

## Technologies Used

- **Python:** The main programming language for the project.
- **Pygame:** Provides a modern graphical user interface (GUI) for displaying the Sudoku board, buttons, and animations.
- **TensorFlow / Keras:** Used for building, training, and running the neural network model that predicts Sudoku cell values.
- **NumPy:** Helps with numerical operations and data manipulation (such as flattening the board).

## Features

- **Complete Board Generation:**  
  A backtracking algorithm generates a fully solved Sudoku board, ensuring a valid starting point for puzzles.
  
- **Puzzle Creation:**  
  The complete board is transformed into a playable puzzle by randomly removing a configurable number of cells.
  
- **AI-Powered Solving:**  
  The project features a neural network model that is trained on generated puzzles to predict cell values.  
  - **Training:**  
    - The training process involves generating many sample puzzles by removing cells from complete boards.
    - Each sample includes an input (the puzzle with empty cells) and a target (the complete, correct board) that is one-hot encoded.
    - The network is then trained using these samples with categorical cross-entropy as the loss function.
  - **Inference & Animation:**  
    - When you click **Solve with AI Visual**, the AI makes predictions for each empty cell.
    - The visualizer animates each cell by first displaying the AI's guess (highlighted in yellow) then revealing the correct value (highlighted in green if the guess was right, or red if not).
  
- **Hybrid Approach for Accuracy:**  
  Even though the neural network is trained to "learn" Sudoku, it does not always produce valid moves on its own. Therefore, a classical backtracking solver is used to compute the true solution. The final board is always correct, and a status message ("Correct" in green or "Incorrect" in red) is displayed below the board after solving.

- **Modern, Professional UI:**  
  The interface uses a clean flat design with smooth animations and intuitive controls for generating new puzzles and watching the AI solve them.

## How It Works

1. **Board Generation:**  
   A complete Sudoku board is created using backtracking. Some numbers are then removed to form a playable puzzle.

2. **Training the AI:**
   - The training script (`train.py`) generates a large dataset of puzzles paired with their solutions.
   - Each puzzle is represented as a flattened 81-number vector.
   - The solution for each puzzle is one-hot encoded (each of the 81 cells is represented by a 9-element vector corresponding to digits 1–9).
   - The neural network (defined in `ai_model.py`) is trained on these inputs and targets.
   - After training, the model is saved as `sudoku_ai_model.h5`.

3. **Visual Solving:**
   - In the Pygame interface (`game.py`), when the user clicks **Solve with AI Visual**, the AI predicts the value for each empty cell.
   - In parallel, the backtracking solver computes the correct solution.
   - The UI then animates each cell: showing the AI’s guess briefly before revealing the correct value.
   - Finally, the complete board is validated, and a final status ("Correct" or "Incorrect") is displayed.
