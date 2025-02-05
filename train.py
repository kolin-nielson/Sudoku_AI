# train.py
import numpy as np
from sudoku import SudokuBoard
from ai_model import build_model
from tensorflow.keras.utils import to_categorical

def board_to_input(board):
    """
    Convert a 9x9 board into a single list (flatten it).
    This is our input for the neural network.
    """
    return np.array(board).flatten().astype(np.float32)

def board_to_target(board):
    """
    Convert a 9x9 board into a target array.
    Each number (1-9) is turned into a one-hot encoded vector.
    """
    flat = np.array(board).flatten() - 1  # Shift numbers down by 1 (so 1 becomes 0, etc.)
    return to_categorical(flat, num_classes=9)

def generate_sample(removals=40):
    # Create a complete board.
    sb = SudokuBoard()
    complete = sb.generate_complete_board()
    # Save a copy of the full solution.
    target_board = [row[:] for row in complete]
    # Remove some numbers to create a puzzle.
    puzzle = sb.remove_numbers(removals)
    return board_to_input(puzzle), board_to_target(target_board)

def generate_dataset(num_samples=1000, removals=40):
    # Generate many samples for training.
    X, y = [], []
    for _ in range(num_samples):
        inp, target = generate_sample(removals)
        X.append(inp)
        y.append(target)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    print("Generating training data...")
    X, y = generate_dataset(num_samples=1000, removals=40)
    print("Data shapes:", X.shape, y.shape)
    
    # Build and show the model.
    model = build_model()
    model.summary()
    
    print("Training model...")
    # Train the model with the generated data.
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
    
    # Save the trained model to a file.
    model.save("sudoku_ai_model.h5")
    print("Model saved as sudoku_ai_model.h5")
