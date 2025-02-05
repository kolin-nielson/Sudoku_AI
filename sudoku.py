# sudoku.py
import random
import copy
from solver import solve_sudoku  # Import the solver to help generate a full board

class SudokuBoard:
    def __init__(self, board=None):
        # Initialize the board as a 9x9 grid filled with 0 (empty cells)
        if board:
            self.board = board
        else:
            self.board = [[0 for _ in range(9)] for _ in range(9)]

    def generate_complete_board(self):
        """Fill the board completely with a valid Sudoku solution."""
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_board()  # Use backtracking to fill the board
        return self.board

    def _fill_board(self):
        # Loop through each cell in the board.
        # If the cell is empty (0), try numbers in random order.
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)  # Randomize order to get variety
                    for number in numbers:
                        if self._is_safe(i, j, number):
                            self.board[i][j] = number
                            if self._fill_board():
                                return True  # If the board is completely filled, we're done
                            self.board[i][j] = 0  # Reset cell if number doesn't lead to a solution
                    return False  # If no number fits, backtrack
        return True

    def _is_safe(self, row, col, num):
        # Check if placing 'num' in (row, col) is allowed by Sudoku rules.
        # Check the row and column.
        if any(self.board[row][x] == num for x in range(9)):
            return False
        if any(self.board[x][col] == num for x in range(9)):
            return False
        # Check the 3x3 square that contains this cell.
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == num:
                    return False
        return True

    def remove_numbers(self, removals=40):
        """
        Remove a number of cells from the complete board to create a puzzle.
        'removals' is the approximate number of cells to clear.
        """
        puzzle = copy.deepcopy(self.board)
        count = 0
        while count < removals:
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            if puzzle[i][j] != 0:
                puzzle[i][j] = 0  # Remove the number
                count += 1
        return puzzle

    def get_board(self):
        # Return the complete board.
        return self.board

# For testing the module by running it directly:
if __name__ == '__main__':
    sb = SudokuBoard()
    sb.generate_complete_board()
    puzzle = sb.remove_numbers(40)
    for row in puzzle:
        print(row)
