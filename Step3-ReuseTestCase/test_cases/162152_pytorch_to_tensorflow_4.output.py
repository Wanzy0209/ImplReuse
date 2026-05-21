import tensorflow as tf
from tensorflow.keras.preprocessing import text

# Define input data (analogous to input_data in the original PyTorch test)
input_data = [
    "The quick brown fox jumps over the lazy dog.",
    "TensorFlow is an end-to-end open source platform for machine learning."
]

# Initialize the tokenizer (analogous to SimpleModel initialization)
# We configure it with specific parameters to mimic the custom setup in the original bug
tokenizer = text.Tokenizer(num_words=1000, oov_token="<OOV>")

# Fit the tokenizer on the data (analogous to moving model to device and setting up DataParallel)
tokenizer.fit_on_texts(input_data)

# Process the data (analogous to the forward pass model(input_data))
sequences = tokenizer.texts_to_sequences(input_data)

# Verify the output (analogous to checking for success)
assert len(sequences) == len(input_data), "Number of sequences should match input"
assert all(isinstance(seq, list) for seq in sequences), "Output should be list of lists"

print("success")