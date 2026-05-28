import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

# Add 'src' directory to Python path to ensure clean imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tokenizer import Tokenizer, load_conversations
from dataset import ConversationDataset
from model import TransformerChatbot

def run_evaluation():
    print("=== STAGE 13: CHATBOT MODEL EVALUATION ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load vocabulary and initialize Tokenizer
    vocab_path = "vocab.json"
    if not os.path.exists(vocab_path):
        print("[!] Error: 'vocab.json' not found. You must train the model first by running 'python train.py'!")
        return
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    tokenizer = Tokenizer()
    tokenizer.vocab = vocab
    tokenizer.inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    # 2. Re-create Model Architecture & Load weights
    model_path = "chatbot_model.pt"
    if not os.path.exists(model_path):
        print("[!] Error: 'chatbot_model.pt' not found. You must train the model first!")
        return
        
    model = TransformerChatbot(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_len=32
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Put in evaluation mode (deactivates dropout)
    
    # 3. Load dataset
    data_path = "data/conversations.txt"
    pairs = load_conversations(data_path)
    dataset = ConversationDataset(pairs=pairs, tokenizer=tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 4. Evaluation metrics
    # ignore_index=-100 ensures we only evaluate on the Bot's response tokens (ignoring User prompt and PAD tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    print("\n[*] Running evaluation over all dataset samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Create causal mask
            seq_len = input_ids.size(1)
            mask_template = torch.tril(torch.ones(seq_len, seq_len, device=device))
            causal_mask = torch.zeros(seq_len, seq_len, device=device).masked_fill(mask_template == 0, float('-inf'))
            
            # Forward pass
            logits = model(input_ids, mask=causal_mask)
            
            # Shift for next-token prediction evaluation
            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            
            # Compute cross entropy loss (average across tokens in this batch)
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item() * (shift_labels != -100).sum().item()
            
            # Calculate accuracy:
            # Get the greedy prediction (index with max logit)
            predictions = torch.argmax(shift_logits, dim=-1)
            
            # We only count predictions where labels are not masked (i.e., not -100)
            valid_mask = (shift_labels != -100)
            correct_predictions = (predictions == shift_labels) & valid_mask
            
            correct_tokens += correct_predictions.sum().item()
            total_tokens += valid_mask.sum().item()
            
    # Compute final metrics
    # Average Loss: total loss / total evaluated tokens
    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    
    # Perplexity (PPL): e^(avg_loss)
    perplexity = math.exp(avg_loss) if total_tokens > 0 else 0.0
    
    # Accuracy: correct tokens / total tokens
    accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    
    # 5. Display Results
    print("\n" + "=" * 50)
    print("              EVALUATION REPORT")
    print("=" * 50)
    print(f"- Total Evaluated Tokens (Bot text): {total_tokens}")
    print(f"- Correct Token Predictions:         {correct_tokens}")
    print(f"- Average Cross-Entropy Loss:        {avg_loss:.4f}")
    print(f"- Model Perplexity (PPL):            {perplexity:.4f}")
    print(f"- Next-Token Prediction Accuracy:    {accuracy * 100:.2f}%")
    print("=" * 50)
    
    if accuracy > 0.85:
        print("[*] Mentor Note: Outstanding! The model has successfully learned")
        print("    to memorize and predict the correct chatbot responses.")
    else:
        print("[*] Mentor Note: Keep training! Increase epochs or adjust learning rates.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_evaluation()
