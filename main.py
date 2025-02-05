# main.py
from sudoku_env import SudokuEnv
from tensorflow.keras.models import load_model
import numpy as np

def run_demo():
    env = SudokuEnv(removals=40)
    model = load_model("sudoku_dqn_model.h5")
    state = env.reset()
    state = state / 9.0
    done = False
    while not done:
        q_values = model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        state, reward, done, _ = env.step(action)
        state = state / 9.0
    print("Final board:")
    env.render()
    
if __name__ == "__main__":
    run_demo()
