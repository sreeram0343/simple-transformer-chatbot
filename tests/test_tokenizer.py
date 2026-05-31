import unittest
from src.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.sample_texts = ["Hello world", "How are you"]
        self.tokenizer.train(self.sample_texts)

    def test_vocab_size(self):
        # 4 special tokens + "hello", "world", "how", "are", "you"
        self.assertEqual(len(self.tokenizer.vocab), 9)

    def test_encode_decode(self):
        text = "Hello how are you"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(text.lower(), decoded)

    def test_unknown_token(self):
        text = "Unknown word test"
        encoded = self.tokenizer.encode(text)
        # "word" and "test" are unknown
        self.assertEqual(encoded[0], self.tokenizer.vocab["hello"])
        self.assertEqual(encoded[1], self.tokenizer.vocab["<UNK>"])
        self.assertEqual(encoded[2], self.tokenizer.vocab["<UNK>"])

    def test_eos_token(self):
        text = "Hello"
        encoded = self.tokenizer.encode(text, add_eos=True)
        self.assertEqual(encoded[-1], self.tokenizer.vocab["<EOS>"])

    def test_special_tokens_skip(self):
        text = "Hello <PAD> world"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded, skip_special=True)
        self.assertEqual(decoded, "hello world")

if __name__ == "__main__":
    unittest.main()
