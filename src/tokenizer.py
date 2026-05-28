from collections import Counter
import os

class Tokenizer:
    """
    A simple word-level Tokenizer designed for our custom chatbot.
    
    Special Tokens:
    - <PAD>: Used to make sequences in a batch the same length (ID: 0).
    - <UNK>: Represents words that are not in our vocabulary (ID: 1).
    - <SEP>: Separates the User prompt from the Bot response (ID: 2).
    - <EOS>: Signals the End of Sentence/Response (ID: 3).
    """
    def __init__(self):
        # 1. Initialize vocabulary with special tokens
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SEP>": 2,
            "<EOS>": 3
        }
        # Inverse vocabulary to easily decode token IDs back into words
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, texts: list[str]):
        """
        Builds the vocabulary from a list of sentence strings.
        """
        counter = Counter()
        for text in texts:
            # Tokenize by splitting on spaces and lowercasing
            words = text.lower().split()
            counter.update(words)
            
        # Assign a unique sequential integer ID to every unique word
        for word in counter:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                
        # Re-build inverse vocabulary for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """
        Converts a raw text string into a list of token IDs.
        """
        words = text.lower().split()
        token_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        
        if add_eos:
            token_ids.append(self.vocab["<EOS>"])
            
        return token_ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """
        Converts a list of token IDs back into a readable string.
        """
        words = []
        for tid in token_ids:
            word = self.inv_vocab.get(tid, "<UNK>")
            
            # Skip special tokens if requested, except for displaying details in debugging
            if skip_special and word in ["<PAD>", "<UNK>", "<SEP>", "<EOS>"]:
                continue
                
            words.append(word)
            
        return " ".join(words)


def load_conversations(file_path: str) -> list[tuple[str, str]]:
    """
    Loads conversation pairs from data/conversations.txt.
    Format is expected to be: user_prompt|bot_response
    """
    pairs = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            user, bot = line.split("|")
            pairs.append((user.strip(), bot.strip()))
            
    return pairs


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== TOKENIZER & DATASET DEMO ===")
    
    # 1. Load data
    data_path = "data/conversations.txt"
    pairs = load_conversations(data_path)
    print(f"\nLoaded {len(pairs)} conversation pairs from '{data_path}'")
    print(f"Sample pair 0: User: '{pairs[0][0]}' | Bot: '{pairs[0][1]}'")
    
    # Extract flat list of sentences to train tokenizer
    flat_texts = []
    for user, bot in pairs:
        flat_texts.append(user)
        flat_texts.append(bot)
        
    # 2. Train Tokenizer
    tokenizer = Tokenizer()
    tokenizer.train(flat_texts)
    print(f"\nTokenizer Vocabulary Size: {len(tokenizer.vocab)}")
    print(f"Special Tokens: {list(tokenizer.vocab.keys())[:4]}")
    
    # 3. Encode & Decode Test
    sample = "Hello how are you"
    encoded = tokenizer.encode(sample, add_eos=True)
    decoded = tokenizer.decode(encoded, skip_special=False)
    
    print(f"\nTest Encoding:")
    print(f"- Text: '{sample}'")
    print(f"- Encoded: {encoded}")
    print(f"- Decoded (with specials): '{decoded}'")