import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add 'src' directory to Python path to ensure clean imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tokenizer import Tokenizer, load_conversations
from dataset import ConversationDataset
from model import TransformerChatbot

def train_chatbot():
    print("=== STARTING TRANSFORMER CHATBOT TRAINING ===")
    
    # -------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -------------------------------------------------------------
    # Write device-agnostic code (runs on GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device detected: {device}")
    
    # Hyperparameters (Optimized for small-scale learning)
    epochs = 40
    batch_size = 2
    learning_rate = 0.001
    
    # Model dimensions
    d_model = 32         # Dimension of token embedding space
    num_heads = 4        # Number of parallel attention heads
    d_ff = 128           # Dimension of FFN hidden layer (4x d_model)
    num_layers = 2       # Number of stacked Transformer Blocks
    max_len = 32         # Maximum sequence length supported
    
    print("\nModel Parameters:")
    print(f"- Embed Dim (d_model): {d_model}")
    print(f"- Attention Heads: {num_heads}")
    print(f"- Stacked Layers: {num_layers}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Epochs: {epochs}")
    
    # -------------------------------------------------------------
    # 2. DATA PREPARATION (STAGE 5)
    # -------------------------------------------------------------
    data_path = "data/conversations.txt"
    print(f"\n[*] Loading conversations from '{data_path}'...")
    pairs = load_conversations(data_path)
    
    # Collect all user and bot sentences to build the tokenizer vocabulary
    flat_texts = []
    for user, bot in pairs:
        flat_texts.append(user)
        flat_texts.append(bot)
        
    tokenizer = Tokenizer()
    tokenizer.train(flat_texts)
    vocab_size = len(tokenizer.vocab)
    print(f"[+] Tokenizer Vocabulary trained. Size: {vocab_size}")
    
    # Save vocabulary to JSON for later decoding during inference (Stage 9)
    vocab_save_path = "vocab.json"
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=4)
    print(f"[+] Tokenizer vocabulary saved to '{vocab_save_path}'")
    
    # Initialize PyTorch Dataset & DataLoader
    dataset = ConversationDataset(pairs=pairs, tokenizer=tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[+] PyTorch DataLoader constructed with {len(dataloader)} batches.")
    
    # -------------------------------------------------------------
    # 3. INITIALIZE MODEL & OPTIMIZER (STAGE 6)
    # -------------------------------------------------------------
    model = TransformerChatbot(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    )
    model.to(device)  # Ship model weights to CPU or GPU memory
    print("[+] Model instantiated and transferred to device.")
    
    # AdamW is the standard modern optimizer for Transformers (incorporates weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # -------------------------------------------------------------
    # 4. TRAINING LOOP & LOSS COMPUTATION (STAGES 6, 7, 8)
    # -------------------------------------------------------------
    # Loss Function (Stage 7): CrossEntropyLoss
    # ignore_index=-100 specifies that targets labeled -100 will NOT contribute to the loss computation.
    # This ensures the model is ONLY trained to predict the Bot response, ignoring User prompts and Padding.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()  # Put the model in training mode (activates dropout)
    
    print("\n[*] Training starting...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        for batch in dataloader:
            # Transfer input IDs and target labels to active device (CPU/GPU)
            # Shapes: (batch_size, max_len)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Create a causal look-ahead mask for this sequence length
            seq_len = input_ids.size(1)
            mask_template = torch.tril(torch.ones(seq_len, seq_len, device=device))
            causal_mask = torch.zeros(seq_len, seq_len, device=device).masked_fill(mask_template == 0, float('-inf'))
            
            # Step A: Forward Pass
            # logits shape: (batch_size, seq_len, vocab_size)
            logits = model(input_ids, mask=causal_mask)
            
            # Step B: Shift Logits and Targets (Stage 7 - Next-Token Prediction alignment)
            # In autoregressive models, we predict the next token:
            # - input at index t predicts label at index t+1.
            # So, we shift inputs (logits) by removing the last token: logits[:, :-1, :]
            # And shift targets (labels) by removing the first token: labels[:, 1:]
            # Then flatten them to feed into PyTorch's CrossEntropyLoss
            # Shifted shapes:
            # - Shifted Logits: (batch_size, seq_len - 1, vocab_size) -> (batch_size * (seq_len - 1), vocab_size)
            # - Shifted Labels: (batch_size, seq_len - 1) -> (batch_size * (seq_len - 1))
            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            
            # Compute loss
            loss = criterion(shift_logits, shift_labels)
            
            # Step C: Backpropagation (Stage 8)
            # 1. Clear existing gradients from the optimizer buffer
            optimizer.zero_grad()
            # 2. Backward pass: computes the derivative of the loss with respect to all model parameters (gradients)
            loss.backward()
            # 3. Parameter update: adjust weights slightly in the opposite direction of the gradients
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate and print average epoch loss
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Average Loss: {avg_loss:.4f}")
            
    # -------------------------------------------------------------
    # 5. SAVE MODEL WEIGHTS (STAGE 9)
    # -------------------------------------------------------------
    model_save_path = "chatbot_model.pt"
    # torch.save serializes the model's state_dict (weights/parameters) into a file.
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[+] Training Complete! Model weights successfully saved to '{model_save_path}'")
    print("=============================================\n")

if __name__ == "__main__":
    train_chatbot()
