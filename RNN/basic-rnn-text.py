import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
text_data = "This is an example of a text sequence. We want to predict the next word in the sequence."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences([text_data])[0]

# Generate input sequences and labels
input_sequences = []
for i in range(1, len(sequences)):
    n_gram_sequence = sequences[:i+1]
    input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1))
model.add(SimpleRNN(units=100, activation='relu'))
model.add(Dense(units=total_words, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

# Generate a new sequence for testing
input_text = "This is an example"
input_sequence = tokenizer.texts_to_sequences([input_text])[0]
input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length-1, padding='pre')

# Make predictions using the trained model
predicted_index = np.argmax(model.predict(input_sequence), axis=-1)
predicted_word = tokenizer.index_word[predicted_index[0]]

print("Input Sequence:")
print(input_text)
print("Predicted Next Word:")
print(predicted_word)
