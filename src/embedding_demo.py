import torch
import torch.nn as nn
from collections import Counter

# ==========================================
# 1. TOKENIZER CLASS
# ==========================================
class Tokenizer:
    def __init__(self):
        # Initialize with special tokens
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1
        }
    
    def train(self, texts):
        """Builds the vocabulary from a list of sentences."""
        counter = Counter()
        
        # Count all unique words
        for text in texts:
            counter.update(text.lower().split())
            
        # Assign an ID to every unique word
        for word in counter:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def encode(self, text):
        """Converts a sentence string into a list of integer IDs."""
        # If a word isn't in the vocab, use the <UNK> ID (1)
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.lower().split()]


# ==========================================
# 2. MAIN EXECUTION (THE PIPELINE)
# ==========================================
if __name__ == "__main__":
    
    # --- Step 1: Prepare the Data ---
    print("--- STEP 1: TRAINING THE TOKENIZER ---")
    dataset = [
        "Hello there",
        "Hi how can I help",
        "I need to learn embeddings",
        "This is a great start"
    ]
    
    tokenizer = Tokenizer()
    tokenizer.train(dataset)
    
    print(f"Vocabulary Size: {len(tokenizer.vocab)}")
    print(f"Vocabulary: {tokenizer.vocab}\n")


    # --- Step 2: Encode a Sentence ---
    print("--- STEP 2: ENCODING TEXT ---")
    sample_sentence = "Hello how can I learn"
    
    # Convert text to Python list of IDs
    token_ids = tokenizer.encode(sample_sentence)
    
    print(f"Original Text: '{sample_sentence}'")
    print(f"Token IDs: {token_ids}\n")


    # --- Step 3: Generate Embeddings ---
    print("--- STEP 3: PYTORCH EMBEDDINGS ---")
    
    # Create the embedding layer
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 4 # Kept small (4) so it prints neatly on your screen
    
    embedding_layer = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim
    )

    # Convert the Python list into a PyTorch Tensor
    input_tensor = torch.tensor(token_ids)
    
    # Pass the IDs through the layer to get the rich vectors
    embedded_output = embedding_layer(input_tensor)

    print(f"Input Tensor:\n{input_tensor}\n")
    print(f"Output Embeddings:\n{embedded_output}\n")
    print(f"Final Tensor Shape: {embedded_output.shape}")