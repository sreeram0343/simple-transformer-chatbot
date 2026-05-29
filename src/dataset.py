import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, load_conversations

class ConversationDataset(Dataset):
    """Custom PyTorch Dataset for training a Decoder-only Transformer Chatbot.

    This dataset handles the concatenation of user prompts and bot
    responses, applies causal masking by setting target labels for user
    input to -100 (ignored by CrossEntropyLoss), and pads sequences to
    a fixed maximum length.

    Attributes:
        pairs (list[tuple[str, str]]): List of (user_prompt, bot_response) strings.
        tokenizer (Tokenizer): A trained Tokenizer instance.
        max_len (int): The maximum sequence length.
    """
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int = 16):
        """Initializes the ConversationDataset.

        Args:
            pairs (list[tuple[str, str]]): List of (user_prompt, bot_response) strings.
            tokenizer (Tokenizer): An initialized and trained Tokenizer instance.
            max_len (int): The maximum sequence length to pad/truncate to.
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the total number of conversation pairs in the dataset.

        Returns:
            int: The dataset length.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves a single conversation item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'labels' tensors.
        """
        user, bot = self.pairs[idx]
        
        # 1. Encode user prompt and bot response
        user_ids = self.tokenizer.encode(user)
        bot_ids = self.tokenizer.encode(bot, add_eos=True)  # Bot response ends with <EOS>
        
        # 2. Combine into a single sequence
        # Format: User_Tokens + <SEP> + Bot_Tokens
        sep_id = self.tokenizer.vocab["<SEP>"]
        combined_ids = user_ids + [sep_id] + bot_ids
        
        # 3. Create the targets (labels)
        # We mask out the user's prompt by assigning a label of -100.
        # Length of user prompt + separator:
        user_prompt_len = len(user_ids) + 1  
        
        # Targets: [-100, -100, ..., bot_id_1, bot_id_2, ..., EOS_id]
        labels = [-100] * user_prompt_len + bot_ids
        
        # 4. Truncate if the sequence exceeds max_len
        if len(combined_ids) > self.max_len:
            combined_ids = combined_ids[:self.max_len]
            labels = labels[:self.max_len]
            
        # 5. Pad both sequences to reach max_len
        # We pad inputs with self.tokenizer.vocab["<PAD>"] (ID 0)
        # We pad targets with -100 (so we don't calculate loss on padding tokens either!)
        padding_length = self.max_len - len(combined_ids)
        
        pad_id = self.tokenizer.vocab["<PAD>"]
        input_ids = combined_ids + [pad_id] * padding_length
        target_labels = labels + [-100] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_labels, dtype=torch.long)
        }


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 5 DEMO: DATASET & LABELS MASKING ===")
    
    # 1. Load conversations and train tokenizer
    data_path = "data/conversations.txt"
    pairs = load_conversations(data_path)
    
    flat_texts = [text for pair in pairs for text in pair]
    tokenizer = Tokenizer()
    tokenizer.train(flat_texts)
    
    # 2. Instantiate Dataset
    max_len = 12  # Let's keep it small for visualization
    dataset = ConversationDataset(pairs=pairs, tokenizer=tokenizer, max_len=max_len)
    
    print(f"\nCreated dataset with {len(dataset)} items. Max Length = {max_len}")
    
    # 3. Inspect a single sample in detail
    sample_idx = 2
    sample = dataset[sample_idx]
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    print(f"\nInspecting Sample #{sample_idx}:")
    print(f"Original User: '{pairs[sample_idx][0]}'")
    print(f"Original Bot:  '{pairs[sample_idx][1]}'")
    
    print(f"\nInput IDs Tensor:   {input_ids.tolist()}")
    print(f"Labels Mask Tensor: {labels.tolist()}")
    
    # Let's align them visually to see the masking
    print("\nVisual alignment of Inputs and Target Labels:")
    print(f"{'Token (Word)':<15} | {'Input ID':<8} | {'Target Label (Shifted in loss)':<15}")
    print("-" * 50)
    for i in range(max_len):
        tid = input_ids[i].item()
        lid = labels[i].item()
        word = tokenizer.inv_vocab.get(tid, "<UNK>")
        print(f"{word:<15} | {tid:<8} | {lid:<15}")
        
    # 4. Create a PyTorch DataLoader
    # Handles batching, shuffling, and multi-threading
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nDataLoader initialized (Batch Size: {batch_size})")
    first_batch = next(iter(dataloader))
    print(f"Batch Input Shape:  {first_batch['input_ids'].shape}")
    print(f"Batch Labels Shape: {first_batch['labels'].shape}")
    
    # 5. Verify the correctness of target labels
    print("\n--- Verifying Label Mask Correctness ---")
    assert input_ids.shape == (max_len,), "Input tensor shape is wrong!"
    assert labels.shape == (max_len,), "Label tensor shape is wrong!"
    # Verify that first token of bot response in input matches first non-negative label in targets
    first_non_masked_idx = (labels != -100).nonzero(as_tuple=True)[0][0].item()
    assert input_ids[first_non_masked_idx].item() == labels[first_non_masked_idx].item(), "Labels do not match input tokens!"
    print("SUCCESS: Dataset masking and sequence padding are perfectly correct!")
