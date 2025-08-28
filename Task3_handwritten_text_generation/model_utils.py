import tensorflow as tf
import numpy as np
import json
import os

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Builds the RNN model using the Keras Functional API."""
    inputs = tf.keras.Input(batch_shape=[batch_size, None])
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform',
                            name='gru_layer')(x)  # Added name for easy access
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def generate_text(model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0):
    """Generates text using the trained model."""
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    # Access the specific GRU layer to reset its states
    model.get_layer('gru_layer').reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

def get_latest_checkpoint(checkpoint_dir):
    """Finds the latest .weights.h5 checkpoint file."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    if not checkpoints:
        return None
    checkpoints.sort()
    return os.path.join(checkpoint_dir, checkpoints[-1])

def load_vocab(vocab_file='vocab.json'):
    """Loads the vocabulary and creates the char2idx and idx2char mappings."""
    with open(vocab_file, 'r') as f:
        char2idx = json.load(f)
    idx2char_map = {idx: char for char, idx in char2idx.items()}
    idx2char = np.array([char for idx, char in sorted(idx2char_map.items())])
    vocab_size = len(idx2char)
    return char2idx, idx2char, vocab_size