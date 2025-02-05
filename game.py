# game.py
import pygame
import numpy as np
import copy
from sudoku import SudokuBoard
from solver import solve_sudoku, is_valid  # 'is_valid' is used for checking the final board
from tensorflow.keras.models import load_model

# Define constants for the window size and board.
WIDTH, HEIGHT = 540, 750  # Extra space for buttons and status message.
GRID_SIZE = 540
CELL_SIZE = GRID_SIZE // 9

# Define colors (using a modern flat design).
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
        # Initialize the Pygame window.
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Sudoku AI Visualizer")
        self.font = pygame.font.SysFont("Helvetica", 36)
        self.small_font = pygame.font.SysFont("Helvetica", 20)
        self.status_font = pygame.font.SysFont("Helvetica", 28, bold=True)
        self.clock = pygame.time.Clock()

        # Try to load the AI model (trained neural network).
        try:
            self.ai_model = load_model("sudoku_ai_model.h5")
            self.ai_available = True
        except Exception as e:
            print("AI model not found or failed to load:", e)
            self.ai_available = False

        self.new_board()

        # These variables help with the animation during solving.
        self.solving_visual = False
        self.visual_steps = []  # List of steps for each cell (with AI guess and true value).
        self.last_update_time = 0
        self.update_interval = 600  # Time (ms) between animation steps

        # After solving, this will store the final message (Correct or Incorrect).
        self.final_status = None

    def new_board(self):
        # Create a new complete board and then remove numbers to make a puzzle.
        sb = SudokuBoard()
        sb.generate_complete_board()
        self.solution_complete = [row[:] for row in sb.get_board()]
        self.board = sb.remove_numbers(removals=40)
        # Keep a copy of the original puzzle to know which numbers were given.
        self.original_puzzle = copy.deepcopy(self.board)
        self.selected = None
        self.solving_visual = False
        self.visual_steps = []
        self.final_status = None

    def draw_grid(self):
        # Fill the window with the background color.
        self.screen.fill(BG_COLOR)
        # Draw each cell as a white box.
        for i in range(9):
            for j in range(9):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, CELL_BG_COLOR, rect)
        # Draw the thick and thin lines to show the grid.
        for i in range(10):
            thickness = 4 if i % 3 == 0 else 1
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), thickness)
            pygame.draw.line(self.screen, GRID_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), thickness)

    def draw_numbers(self):
        # Draw the numbers on the board.
        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                if num != 0:
                    # If the number is from the original puzzle, show it in bold.
                    is_given = self.original_puzzle[i][j] != 0
                    font = pygame.font.SysFont("Helvetica", 36, bold=is_given)
                    text = font.render(str(num), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
                    self.screen.blit(text, text_rect)

    def draw_buttons(self):
        # Draw two buttons: one for a new board and one to start the solving animation.
        self.draw_button("New Board", (20, GRID_SIZE + 20, 150, 45))
        self.draw_button("Solve with AI Visual", (WIDTH - 210, GRID_SIZE + 20, 190, 45))

    def draw_button(self, text, rect):
        # Draw a single button with hover effects.
        x, y, w, h = rect
        mouse = pygame.mouse.get_pos()
        is_hover = x <= mouse[0] <= x + w and y <= mouse[1] <= y + h
        color = BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        txt = self.small_font.render(text, True, (255, 255, 255))
        txt_rect = txt.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(txt, txt_rect)

    def draw_status(self):
        # If the final status is set, draw the "Correct" or "Incorrect" message below the grid.
        if self.final_status:
            text, color = self.final_status
            status_text = self.status_font.render(text, True, color)
            status_rect = status_text.get_rect(center=(WIDTH // 2, GRID_SIZE + 90))
            self.screen.blit(status_text, status_rect)

    def get_cell_from_pos(self, pos):
        # Convert a mouse click position into a board cell (row, column).
        x, y = pos
        if x < GRID_SIZE and y < GRID_SIZE:
            return y // CELL_SIZE, x // CELL_SIZE
        return None

    def handle_mouse(self, pos):
        # Process mouse clicks.
        x, y = pos
        if y > GRID_SIZE:
            # If the click is below the board, check which button was clicked.
            if 20 <= x <= 170 and GRID_SIZE + 20 <= y <= GRID_SIZE + 65:
                self.new_board()  # New Board button
            elif WIDTH - 210 <= x <= WIDTH - 20 and GRID_SIZE + 20 <= y <= GRID_SIZE + 65:
                self.start_visual_ai_solving()  # Solve button
        else:
            # If the click is on the board, optionally allow selection.
            cell = self.get_cell_from_pos(pos)
            if cell:
                self.selected = cell

    def handle_key(self, key):
        # Allow the user to type a number into a selected cell if it wasn't given originally.
        if self.selected and key in range(pygame.K_1, pygame.K_9 + 1):
            i, j = self.selected
            if self.original_puzzle[i][j] == 0:
                self.board[i][j] = key - pygame.K_0
            self.selected = None

    def start_visual_ai_solving(self):
        """
        This function starts the solving animation:
          1. The AI model makes a guess for each empty cell.
          2. The true solution is computed using the backtracking solver.
          3. Each cell is animated: first showing the AI guess, then the correct value.
        """
        if not self.ai_available:
            print("AI model not available.")
            return
        # Prepare the board input for the AI (flatten the board into one list).
        board_flat = np.array(self.board).flatten().astype(np.float32).reshape(1, -1)
        prediction = self.ai_model.predict(board_flat)
        pred_digits = prediction.argmax(axis=-1).reshape(81,) + 1

        # Make a copy of the board and solve it to get the correct values.
        board_copy = [row[:] for row in self.board]
        if not solve_sudoku(board_copy):
            print("No solution found by the solver!")
            return

        # For every cell that was empty originally, store a step with the AI guess and true value.
        self.visual_steps = []
        for idx in range(81):
            i, j = divmod(idx, 9)
            if self.original_puzzle[i][j] == 0:
                ai_val = int(pred_digits[idx])
                correct_val = board_copy[i][j]
                is_correct = (ai_val == correct_val)
                self.visual_steps.append((i, j, ai_val, correct_val, is_correct))
        self.solving_visual = True
        self.last_update_time = pygame.time.get_ticks()
        self.final_status = None  # Reset any previous final message
        print("Visual solving started with", len(self.visual_steps), "steps.")

    def update_visual_solving(self):
        """
        This function handles the animation.
        It processes one cell at a time, showing the AI's guess briefly,
        then replacing it with the correct value.
        """
        if not self.solving_visual:
            return
        now = pygame.time.get_ticks()
        if now - self.last_update_time > self.update_interval and self.visual_steps:
            # Get the next step.
            i, j, ai_val, correct_val, is_correct = self.visual_steps.pop(0)
            # First, show the AI's guessed value.
            self.board[i][j] = ai_val
            self.draw_grid()
            self.draw_numbers()
            self.draw_buttons()
            self.highlight_cell(i, j, HIGHLIGHT_COLOR, 4)
            pygame.display.update()
            pygame.time.delay(300)  # Pause briefly to show the guess

            # Now, update the cell with the correct value.
            self.board[i][j] = correct_val
            color = CORRECT_HIGHLIGHT if is_correct else INCORRECT_HIGHLIGHT
            self.highlight_cell(i, j, color, 4)
            self.last_update_time = now

        # When all steps are finished, check if the board is correct.
        if not self.visual_steps and self.solving_visual:
            self.solving_visual = False
            self.final_status = self.check_board_status()

    def check_board_status(self):
        """
        Check the completed board:
          - Each row, column, and 3x3 block must have unique digits 1-9.
        Return a message and a color (green for correct, red for incorrect).
        """
        # Check each row.
        for i in range(9):
            row = [self.board[i][j] for j in range(9)]
            if len(set(row)) != 9:
                return ("Incorrect", INCORRECT_HIGHLIGHT)
        # Check each column.
        for j in range(9):
            col = [self.board[i][j] for i in range(9)]
            if len(set(col)) != 9:
                return ("Incorrect", INCORRECT_HIGHLIGHT)
        # Check each 3x3 block.
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
        # If a cell is selected by the user, draw a highlight.
        if self.selected:
            i, j = self.selected
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 3)

    def draw_all(self):
        # Draw the board, numbers, buttons, selected cell, and status message.
        self.draw_grid()
        self.draw_numbers()
        self.highlight_selected()
        self.draw_buttons()
        self.draw_status()

    def run(self):
        # The main game loop.
        running = True
        while running:
            self.clock.tick(30)  # Limit to 30 frames per second
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            
            # Update the solving animation if it is running.
            self.update_visual_solving()
            self.draw_all()
            pygame.display.update()
        pygame.quit()

if __name__ == '__main__':
    game = SudokuGame()
    game.run()
