# sudoku_env.py
import gym
import numpy as np
import copy
from gym import spaces
from sudoku import SudokuBoard

class SudokuEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, removals=40):
        super(SudokuEnv, self).__init__()
        self.removals = removals
        self.action_space = spaces.Discrete(9 * 9 * 9)
        self.observation_space = spaces.Box(low=0, high=9, shape=(81,), dtype=np.int32)
        self.reset()
        
    def reset(self):
        sb = SudokuBoard()
        sb.generate_complete_board()
        self.solution = [row[:] for row in sb.get_board()]
        self.board = sb.remove_numbers(self.removals)
        self.original = copy.deepcopy(self.board)
        self.done = False
        return np.array(self.board).flatten()
    
    def step(self, action):
        if self.done:
            return np.array(self.board).flatten(), 0, True, {}
            
        cell_index = action // 9
        digit = (action % 9) + 1
        i = cell_index // 9
        j = cell_index % 9
        
        reward = -1  # Default penalty
        
        if self.original[i][j] != 0:
            reward = -1  # Penalize modifying fixed cells
        else:
            if self._is_valid_move(i, j, digit):
                self.board[i][j] = digit
                # Check if the placed digit matches the solution
                if digit == self.solution[i][j]:
                    reward = 1  # Correct digit
                else:
                    reward = -1  # Valid but incorrect
                if self._is_board_full():
                    if self._check_board():
                        reward = 100  # Correct solution
                    else:
                        reward = -10  # Incorrect solution
                    self.done = True
            else:
                reward = -1  # Invalid move
        
        return np.array(self.board).flatten(), reward, self.done, {}
    
    def _is_valid_move(self, i, j, digit):
        if digit in self.board[i]:
            return False
        for row in self.board:
            if row[j] == digit:
                return False
        start_row = 3 * (i // 3)
        start_col = 3 * (j // 3)
        for a in range(start_row, start_row + 3):
            for b in range(start_col, start_col + 3):
                if self.board[a][b] == digit:
                    return False
        return True
    
    def _is_board_full(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return False
        return True
    
    def _check_board(self):
        for i in range(9):
            if len(set(self.board[i])) != 9:
                return False
        for j in range(9):
            col = [self.board[i][j] for i in range(9)]
            if len(set(col)) != 9:
                return False
        for box_row in range(3):
            for box_col in range(3):
                nums = []
                for i in range(3):
                    for j in range(3):
                        nums.append(self.board[box_row*3 + i][box_col*3 + j])
                if len(set(nums)) != 9:
                    return False
        return True
    
    def render(self, mode='human'):
        for row in self.board:
            print(row)
        print("")