# sudoku.py
import random
import copy
from solver import solve_sudoku

class SudokuBoard:
    def __init__(self, board=None):
        if board:
            self.board = board
        else:
            self.board = [[0 for _ in range(9)] for _ in range(9)]

    def generate_complete_board(self):
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_board()
        return self.board

    def _fill_board(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)
                    for number in numbers:
                        if self._is_safe(i, j, number):
                            self.board[i][j] = number
                            if self._fill_board():
                                return True
                            self.board[i][j] = 0
                    return False
        return True

    def _is_safe(self, row, col, num):
        if any(self.board[row][x] == num for x in range(9)):
            return False
        if any(self.board[x][col] == num for x in range(9)):
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == num:
                    return False
        return True

    def remove_numbers(self, removals=40):
        puzzle = copy.deepcopy(self.board)
        count = 0
        while count < removals:
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            if puzzle[i][j] != 0:
                puzzle[i][j] = 0
                count += 1
        return puzzle

    def get_board(self):
        return self.board

if __name__ == '__main__':
    sb = SudokuBoard()
    sb.generate_complete_board()
    puzzle = sb.remove_numbers(40)
    for row in puzzle:
        print(row)
