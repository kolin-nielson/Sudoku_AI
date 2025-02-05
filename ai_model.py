# ai_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    # Define the input layer for a board represented as 81 numbers.
    input_layer = layers.Input(shape=(81,), name="sudoku_input")
    # Add two hidden layers with 256 nodes each.
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.Dense(256, activation='relu')(x)
    # The output layer predicts a probability for each digit in each cell.
    # 81 cells * 9 possible digits = 729 total outputs.
    output = layers.Dense(81 * 9, activation='softmax')(x)
    output = layers.Reshape((81, 9), name="sudoku_output")(output)

    # Create the model and compile it with an optimizer and loss function.
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# If run directly, show the model summary.
if __name__ == '__main__':
    model = build_model()
    model.summary()
