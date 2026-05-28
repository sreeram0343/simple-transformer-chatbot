import sys
import os
import json
import torch

# Add 'src' directory to Python path to ensure clean imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from tokenizer import Tokenizer
from model import TransformerChatbot

class ChatbotPipeline:
    """
    Integrates the full Inference Pipeline, Interactive Chat Interface, 
    and Conversation Memory (Stages 10, 11, 12).
    
    Inference Pipeline (Stage 10):
    1. Encode the user prompt to token IDs and append the <SEP> token.
    2. Repeatedly run the model to predict the logits of the next token.
    3. Use Greedy Search (argmax) to select the most likely token.
    4. Append the predicted token to the input sequence and repeat.
    5. Terminate when <EOS> is generated or max token limit is reached.
    
    Conversation Memory (Stage 12):
    Since Transformers are stateless, they have no built-in memory. To remember context,
    we maintain a history of the conversation: [(User_1, Bot_1), (User_2, Bot_2)].
    On each turn, we concatenate this history:
        User_1 <SEP> Bot_1 <EOS> User_2 <SEP> Bot_2 <EOS> Current_User <SEP>
    This format matches the training dataset exactly, allowing the model to naturally use context!
    """
    def __init__(self, model_path: str = "chatbot_model.pt", vocab_path: str = "vocab.json"):
        # Detect active device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Vocabulary and Initialize Tokenizer
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at '{vocab_path}'. Please run training first!")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        self.tokenizer = Tokenizer()
        self.tokenizer.vocab = vocab
        self.tokenizer.inv_vocab = {v: k for k, v in vocab.items()}
        
        # 2. Reconstruct Model Architecture
        # These hyperparameters MUST match train.py exactly
        d_model = 32
        num_heads = 4
        d_ff = 128
        num_layers = 2
        
        self.model = TransformerChatbot(
            vocab_size=len(self.tokenizer.vocab),
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=32
        )
        
        # 3. Load Trained Model Weights (Stage 9)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model weights not found at '{model_path}'. Please run training first!")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Put the model in evaluation mode (deactivates dropout)
        
        # 4. Initialize Conversation Memory
        self.history = []  # List of (user_message, bot_response) tuples

    def generate_response(self, user_message: str, max_new_tokens: int = 20) -> str:
        """
        Runs the autoregressive inference pipeline with conversation memory.
        """
        sep_id = self.tokenizer.vocab["<SEP>"]
        eos_id = self.tokenizer.vocab["<EOS>"]
        
        # Step A: Build input token sequence with Conversation Memory
        input_ids = []
        for prev_user, prev_bot in self.history:
            # Add past prompt + <SEP> + past response + <EOS>
            input_ids += self.tokenizer.encode(prev_user) + [sep_id] + self.tokenizer.encode(prev_bot, add_eos=True)
            
        # Add current user prompt + <SEP>
        current_prompt_ids = self.tokenizer.encode(user_message) + [sep_id]
        input_ids += current_prompt_ids
        
        # Keep track of where the current bot response begins in our input sequence
        response_start_idx = len(input_ids)
        
        # Step B: Autoregressive Loop
        for _ in range(max_new_tokens):
            # Convert list of token IDs to a PyTorch tensor: shape (1, seq_len)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Create the causal mask for the current sequence length
            seq_len = input_tensor.size(1)
            mask_template = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
            causal_mask = torch.zeros(seq_len, seq_len, device=self.device).masked_fill(mask_template == 0, float('-inf'))
            
            # Pass through the model
            with torch.no_grad():
                # logits shape: (1, seq_len, vocab_size)
                logits = self.model(input_tensor, mask=causal_mask)
                
            # Get the logits of the last token in the sequence (next word logits)
            # shape: (vocab_size,)
            next_token_logits = logits[0, -1, :]
            
            # Greedy Search: pick the token ID with the highest score
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Append the predicted token to the input sequence
            input_ids.append(next_token_id)
            
            # Stop if the model generates the End of Sentence (<EOS>) token
            if next_token_id == eos_id:
                break
                
        # Step C: Extract and decode only the generated response
        generated_token_ids = input_ids[response_start_idx:]
        response_text = self.tokenizer.decode(generated_token_ids, skip_special=True)
        
        # Save the new exchange to memory
        self.history.append((user_message, response_text))
        
        return response_text

    def clear_memory(self):
        """Resets conversation history."""
        self.history = []


# =====================================================================
# INTERACTIVE CHAT INTERFACE (STAGE 11)
# =====================================================================
def run_chat_interface():
    print("\n" + "=" * 60)
    print("         🔮 WELCOME TO THE CUSTOM TRANSFORMER CHATBOT! 🔮")
    print("=" * 60)
    print("[*] Loading chatbot pipeline...")
    
    try:
        pipeline = ChatbotPipeline(model_path="chatbot_model.pt", vocab_path="vocab.json")
        print("[+] Chatbot ready! Actively waiting for your inputs.")
        print("\nCommands:")
        print(" - Type 'exit' to quit the chatbot.")
        print(" - Type 'clear' to wipe conversation memory.")
        print("-" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to load chatbot files: {e}")
        print("[!] Make sure to run 'python train.py' first to train and save the model!")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "exit":
                print("\nMini Bot: Goodbye! Have a great day!")
                break
                
            if user_input.lower() == "clear":
                pipeline.clear_memory()
                print("\n[System] Conversation memory cleared!")
                continue
                
            # Generate and print response
            response = pipeline.generate_response(user_input)
            print(f"Mini Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nMini Bot: Goodbye!")
            break

if __name__ == "__main__":
    run_chat_interface()
