# game.py
import pygame
import numpy as np
import copy
from sudoku import SudokuBoard
from solver import is_valid  # Used later for final board checking
from tensorflow.keras.models import load_model

# Window and board size constants.
WIDTH, HEIGHT = 540, 750  # Extra vertical space for buttons and a status message.
GRID_SIZE = 540
CELL_SIZE = GRID_SIZE // 9

# Define colors using a modern flat design.
BG_COLOR = (248, 248, 248)
GRID_COLOR = (50, 50, 50)
CELL_BG_COLOR = (255, 255, 255)
BUTTON_COLOR = (52, 152, 219)
BUTTON_HOVER_COLOR = (41, 128, 185)
TEXT_COLOR = (44, 62, 80)
HIGHLIGHT_COLOR = (241, 196, 15)
CORRECT_HIGHLIGHT = (39, 174, 96)
INCORRECT_HIGHLIGHT = (192, 57, 43)

class SudokuGame:
    def __init__(self):
        # Initialize the Pygame window, fonts, and clock.
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sudoku AI Visualizer")
        self.font = pygame.font.SysFont("Helvetica", 36)
        self.small_font = pygame.font.SysFont("Helvetica", 20)
        self.status_font = pygame.font.SysFont("Helvetica", 28, bold=True)
        self.clock = pygame.time.Clock()

        # Try to load the trained AI model.
        try:
            self.ai_model = load_model("sudoku_ai_model.h5")
            self.ai_available = True
        except Exception as e:
            print("AI model not found or failed to load:", e)
            self.ai_available = False

        self.new_board()

        # Variables for controlling the iterative solving process.
        self.solving_iterative = False  # True when iterative solving is active.
        self.last_update_time = 0       # Tracks when the last update happened.
        self.update_interval = 600      # Time (in ms) between iterative steps.

        # This variable will hold the final status message ("Correct" or "Incorrect").
        self.final_status = None

    def new_board(self):
        # Create a new complete board, then remove some numbers to make a puzzle.
        sb = SudokuBoard()
        sb.generate_complete_board()
        self.solution_complete = [row[:] for row in sb.get_board()]  # The full solution (for reference only)
        self.board = sb.remove_numbers(removals=40)
        # Save the original puzzle so we know which cells are fixed.
        self.original_puzzle = copy.deepcopy(self.board)
        self.selected = None
        self.solving_iterative = False
        self.final_status = None

    def draw_grid(self):
        # Fill the window with the background color and draw each cell as a white box.
        self.screen.fill(BG_COLOR)
        for i in range(9):
            for j in range(9):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, CELL_BG_COLOR, rect)
        # Draw grid lines (thicker every 3 cells).
        for i in range(10):
            thickness = 4 if i % 3 == 0 else 1
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), thickness)
            pygame.draw.line(self.screen, GRID_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), thickness)

    def draw_numbers(self):
        # Draw numbers on the board.
        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                if num != 0:
                    # Given clues are drawn in bold.
                    is_given = self.original_puzzle[i][j] != 0
                    font = pygame.font.SysFont("Helvetica", 36, bold=is_given)
                    text = font.render(str(num), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2,
                                                      i * CELL_SIZE + CELL_SIZE // 2))
                    self.screen.blit(text, text_rect)

    def draw_buttons(self):
        # Draw two buttons: one to generate a new board and one to start the AI solving process.
        self.draw_button("New Board", (20, GRID_SIZE + 20, 150, 45))
        self.draw_button("Solve with AI", (WIDTH - 210, GRID_SIZE + 20, 190, 45))

    def draw_button(self, text, rect):
        # Draw a button with a hover effect.
        x, y, w, h = rect
        mouse = pygame.mouse.get_pos()
        is_hover = x <= mouse[0] <= x + w and y <= mouse[1] <= y + h
        color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        txt = self.small_font.render(text, True, (255, 255, 255))
        txt_rect = txt.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(txt, txt_rect)

    def draw_status(self):
        # If a final status message exists, display it below the board.
        if self.final_status:
            text, color = self.final_status
            status_text = self.status_font.render(text, True, color)
            status_rect = status_text.get_rect(center=(WIDTH // 2, GRID_SIZE + 90))
            self.screen.blit(status_text, status_rect)

    def get_cell_from_pos(self, pos):
        # Convert a mouse click position into a board cell (row, col).
        x, y = pos
        if x < GRID_SIZE and y < GRID_SIZE:
            return y // CELL_SIZE, x // CELL_SIZE
        return None

    def handle_mouse(self, pos):
        # Process mouse clicks.
        x, y = pos
        if y > GRID_SIZE:
            # If clicked below the board, check if a button was clicked.
            if 20 <= x <= 170 and GRID_SIZE + 20 <= y <= GRID_SIZE + 65:
                self.new_board()  # New Board button
            elif WIDTH - 210 <= x <= WIDTH - 20 and GRID_SIZE + 20 <= y <= GRID_SIZE + 65:
                # Start the iterative solving process.
                self.solving_iterative = True
                self.final_status = None
                # Immediately take one iterative step.
                self.iterative_solve_step()
                self.last_update_time = pygame.time.get_ticks()
        else:
            # If clicked on the board, allow the user to select a cell (for manual editing, if desired).
            cell = self.get_cell_from_pos(pos)
            if cell:
                self.selected = cell

    def handle_key(self, key):
        # Allow the user to type a number into a selected cell (if it’s not a given clue).
        if self.selected and key in range(pygame.K_1, pygame.K_9 + 1):
            i, j = self.selected
            if self.original_puzzle[i][j] == 0:
                self.board[i][j] = key - pygame.K_0
            self.selected = None

    def iterative_solve_step(self):
        """
        This function simulates a human-like, iterative solving step:
          1. The current board (with some cells filled) is fed into the AI model.
          2. For each cell that is not a given clue, the model produces a probability distribution.
          3. We pick the cell with the highest confidence (i.e. the largest softmax probability)
             where the current value is either empty or different from the model's best guess.
          4. We then update that cell with the model's prediction.
          5. This step can update cells in any order, mimicking a human who jumps around the board.
        """
        # Flatten the board and get the model's prediction.
        board_flat = np.array(self.board).flatten().astype(np.float32).reshape(1, -1)
        prediction = self.ai_model.predict(board_flat)
        # Get the predicted digit (and its confidence) for each cell.
        pred_probs = prediction.reshape(81, 9)
        pred_digits = pred_probs.argmax(axis=-1) + 1  # Best guess for each cell.
        confidences = pred_probs.max(axis=-1)         # Confidence (probability) for best guess.

        best_cell = None
        best_confidence = 0

        # Check every cell that is not a given clue.
        for idx in range(81):
            i, j = divmod(idx, 9)
            if self.original_puzzle[i][j] != 0:
                continue  # Skip cells that were given initially.
            # If the cell is empty or its current value differs from the model's guess...
            if self.board[i][j] == 0 or self.board[i][j] != pred_digits[idx]:
                if confidences[idx] > best_confidence:
                    best_confidence = confidences[idx]
                    best_cell = (i, j, pred_digits[idx])

        # If we found a cell to update, do it.
        if best_cell:
            i, j, new_val = best_cell
            self.board[i][j] = new_val
            # For visualization, highlight the cell change.
            self.draw_grid()
            self.draw_numbers()
            self.draw_buttons()
            self.highlight_cell(i, j, HIGHLIGHT_COLOR, 4)
            pygame.display.update()
            pygame.time.delay(300)
        # Else, no changes are suggested by the model.
        # (It may mean the model is confident in every cell.)

    def update_iterative_solving(self):
        """
        This function repeatedly calls the iterative solving step until the board is full.
        It also allows for corrections—if the AI later suggests a different number in a cell,
        that cell can be updated.
        When the board is full, it checks the entire board for correctness.
        """
        if not self.solving_iterative:
            return

        now = pygame.time.get_ticks()
        if now - self.last_update_time > self.update_interval:
            # Take one iterative solving step.
            self.iterative_solve_step()
            self.last_update_time = now

        # Check if the board is completely filled.
        board_full = all(self.board[i][j] != 0 for i in range(9) for j in range(9))
        if board_full:
            # Stop the iterative solving process.
            self.solving_iterative = False
            # Finally, check the board for correctness.
            self.final_status = self.check_board_status()

    def check_board_status(self):
        """
        Check if the board is valid:
          - Each row, column, and 3x3 block must contain unique digits 1-9.
        Returns ("Correct", color) if valid, else ("Incorrect", color).
        """
        # Check rows.
        for i in range(9):
            row = [self.board[i][j] for j in range(9)]
            if len(set(row)) != 9:
                return ("Incorrect", INCORRECT_HIGHLIGHT)
        # Check columns.
        for j in range(9):
            col = [self.board[i][j] for i in range(9)]
            if len(set(col)) != 9:
                return ("Incorrect", INCORRECT_HIGHLIGHT)
        # Check 3x3 blocks.
        for box_row in range(3):
            for box_col in range(3):
                nums = []
                for i in range(3):
                    for j in range(3):
                        nums.append(self.board[box_row * 3 + i][box_col * 3 + j])
                if len(set(nums)) != 9:
                    return ("Incorrect", INCORRECT_HIGHLIGHT)
        return ("Correct", CORRECT_HIGHLIGHT)

    def highlight_cell(self, i, j, color, thickness=4):
        # Draw a colored border around a specific cell.
        rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, thickness)

    def highlight_selected(self):
        # Highlight a cell if the user has selected it.
        if self.selected:
            i, j = self.selected
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 3)

    def draw_all(self):
        # Draw all UI elements: board grid, numbers, buttons, and status message.
        self.draw_grid()
        self.draw_numbers()
        self.highlight_selected()
        self.draw_buttons()
        self.draw_status()

    def run(self):
        # Main game loop.
        running = True
        while running:
            self.clock.tick(30)  # Limit to 30 frames per second.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            
            # If the iterative solving process is active, update it.
            self.update_iterative_solving()
            self.draw_all()
            pygame.display.update()
        pygame.quit()

if __name__ == '__main__':
    game = SudokuGame()
    game.run()
