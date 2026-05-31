import unittest
import torch
import sys
import os

# Add 'src' directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataset import ConversationDataset
from tokenizer import Tokenizer

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.pairs = [("Hi", "Hello"), ("How are you", "I am fine")]
        self.tokenizer.train(["Hi", "Hello", "How are you", "I am fine"])
        self.max_len = 16
        self.dataset = ConversationDataset(self.pairs, self.tokenizer, self.max_len)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_structure(self):
        sample = self.dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertEqual(sample["input_ids"].shape, (self.max_len,))
        self.assertEqual(sample["labels"].shape, (self.max_len,))

    def test_masking_correctness(self):
        # Sample 0: "Hi" (user) | "Hello" (bot)
        # Expected inputs: [Hi, <SEP>, Hello, <EOS>, <PAD>, ...]
        # Expected labels: [-100, -100, Hello, <EOS>, -100, ...]
        sample = self.dataset[0]
        input_ids = sample["input_ids"].tolist()
        labels = sample["labels"].tolist()
        
        # "Hi" is vocab[4], <SEP> is vocab[2]
        self.assertEqual(labels[0], -100) # "Hi" masked
        self.assertEqual(labels[1], -100) # <SEP> masked
        
        # "Hello" is vocab[5], <EOS> is vocab[3]
        self.assertEqual(labels[2], self.tokenizer.vocab["hello"])
        self.assertEqual(labels[3], self.tokenizer.vocab["<EOS>"])
        
        # Padding should be masked
        self.assertEqual(labels[4], -100)

    def test_truncation(self):
        small_max_len = 2
        dataset = ConversationDataset(self.pairs, self.tokenizer, small_max_len)
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].shape, (small_max_len,))
        self.assertEqual(sample["labels"].shape, (small_max_len,))

if __name__ == "__main__":
    unittest.main()
