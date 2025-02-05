# solver.py

def find_empty(board):
    """
    Look for the first empty cell (with a value of 0) in the board.
    Returns the row and column index or None if the board is full.
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, row, col, num):
    """
    Check if placing 'num' at position (row, col) violates Sudoku rules.
    Returns False if it is not allowed.
    """
    # Check row.
    if any(board[row][x] == num for x in range(9)):
        return False
    # Check column.
    if any(board[x][col] == num for x in range(9)):
        return False
    # Check the 3x3 square.
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board):
    """
    Solve the Sudoku puzzle using backtracking.
    The board is changed in-place. Returns True if solved.
    """
    empty = find_empty(board)
    if not empty:
        return True  # Puzzle solved
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True  # If solved further down, return True
            board[row][col] = 0  # Reset cell if not working
    return False

# If run directly, test the solver with a sample puzzle.
if __name__ == '__main__':
    sample_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    if solve_sudoku(sample_puzzle):
        for row in sample_puzzle:
            print(row)
    else:
        print("No solution found.")
