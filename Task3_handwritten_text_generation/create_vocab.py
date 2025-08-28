# create_vocab.py
import json

DATA_PATH = 'data/shakespeare.txt' 

print(f"Loading data from {DATA_PATH} to create vocabulary...")
try:
    text = open(DATA_PATH, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    
    
    char2idx = {char: i for i, char in enumerate(vocab)}
    
    
    with open('vocab.json', 'w') as f:
        json.dump(char2idx, f)
        
    print(f"Successfully created vocab.json with {len(vocab)} unique characters.")
    
except FileNotFoundError:
    print(f"Error: Could not find the dataset at {DATA_path}. Please check the path.")