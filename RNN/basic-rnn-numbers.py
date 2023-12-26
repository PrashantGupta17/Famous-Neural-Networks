import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate some random data for demonstration
np.random.seed(42)
seq_length = 10
num_samples = 1000

# Generate random sequences of numbers between 0 and 1
X = np.random.rand(num_samples, seq_length, 1)
# Sum the values along the sequence axis to create the target output
y = np.sum(X, axis=1)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Generate a new sequence for testing
new_sequence = np.random.rand(1, seq_length, 1)

# Make predictions using the trained model
predicted_output = model.predict(new_sequence)

print("Input Sequence:")
print(new_sequence[0, :, 0])
print("Predicted Output:")
print(predicted_output[0, 0])
