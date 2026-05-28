# Import the counter module from python libraries.
# counter counts how many times a word repeated.

from collections import Counter


def load_conversations(file_path):
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            user, bot = line.strip().split("|")
            texts.append(user)
            texts.append(bot)

    return texts


def build_vocab(texts):
    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    vocab = {
        "<PAD>": 0,           
        "<UNK>": 1
    }

    for word in counter:
        vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    """
    Convert a sentence into token IDs.
    Unknown words become <UNK>.
    """

    tokens = text.lower().split()

    token_ids = []

    for token in tokens:
        token_ids.append(
            vocab.get(token, vocab["<UNK>"])
        )

    return token_ids



if __name__ == "__main__":

    texts = load_conversations("data/conversations.txt")

    vocab = build_vocab(texts)

    print("Vocabulary:")
    print(vocab)

    sentence = "hello how are you"

    encoded = encode(sentence, vocab)

    print("\nEncoded:")
    print(encoded)