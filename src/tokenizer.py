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


if __name__ == "__main__":
    texts = load_conversations("data/conversations.txt")

    vocab = build_vocab(texts)

    print(vocab)