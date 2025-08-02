# text_generator.py (No Comments)

import tensorflow as tf
import numpy as np
import os
import time

DATA_PATH = 'data/shakespeare.txt'
CHECKPOINT_DIR = './training_checkpoints'

SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 20

def load_data(path):
    print(f"Loading data from {path}...")
    try:
        text = open(path, 'rb').read().decode(encoding='utf-8')
        print(f'Length of text: {len(text)} characters')
        return text
    except FileNotFoundError:
        print(f"Error: Dataset not found at {path}")
        return None

def preprocess_data(text):
    print("Preprocessing data...")
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset, vocab, char2idx, idx2char

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Builds the RNN model using the Keras Functional API."""
    print("Building the RNN model...")

    
    inputs = tf.keras.Input(batch_shape=[batch_size, None])

    
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)

    
    x = tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform')(x)

    
    outputs = tf.keras.layers.Dense(vocab_size)(x)

    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, dataset, epochs, checkpoint_dir):
    """Trains the model and saves checkpoints with the correct file extension."""
    print("Starting model training...")
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)

    # Configure checkpoints to save model weights with the required extension
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5") # <-- THE FIX IS HERE
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    return history

def generate_text(model, start_string, char2idx, idx2char):
    """Generates text using the trained model."""
    print("Generating text...")
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    temperature = 1.0
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
        
    return start_string + ''.join(text_generated)

if __name__ == '__main__':
    text = load_data(DATA_PATH)
    if text:
        dataset, vocab, char2idx, idx2char = preprocess_data(text)
        
        vocab_size = len(vocab)
        
        model = build_model(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            rnn_units=RNN_UNITS,
            batch_size=BATCH_SIZE)
            
        
        train_new_model = False 
        
        if train_new_model:
            train_model(model, dataset, EPOCHS, CHECKPOINT_DIR)
        
        # --- Generation Phase ---
        print("\nPreparing model for generation...")

        # Find the latest checkpoint file manually and robustly
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.weights.h5')]
        if not checkpoint_files:
            print("!!! ERROR: No checkpoint files found. Please train the model first by setting train_new_model=True.")
            exit()
        
        # Sort the files to find the one with the highest epoch number
        checkpoint_files.sort()
        latest_checkpoint_file = checkpoint_files[-1]
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint_file)

        print(f"Loading weights from the latest checkpoint: {latest_checkpoint_path}")

        # Build a new model with a batch size of 1 for single-step prediction
        model_for_generation = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
        
        # Load the weights from our manually found checkpoint
        model_for_generation.load_weights(latest_checkpoint_path)
        model_for_generation.build(tf.TensorShape([1, None]))
        
        print("\n--- Model Ready for Generation ---")
        start_seed = input("Enter a starting string (e.g., 'ROMEO:' or 'JULIET:'):\n> ")
        generated_text = generate_text(model_for_generation, start_seed, char2idx, idx2char)
        
        print("\n--- Generated Text ---")
        print(generated_text)