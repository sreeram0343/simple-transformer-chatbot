import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock

class TransformerChatbot(nn.Module):
    """
    A complete Decoder-only GPT-style Transformer Chatbot.
    
    Why it exists:
    This class ties together all the individual components we built:
    1. Token Embeddings: Maps token IDs to vectors.
    2. Positional Encoding: Injects spatial position information.
    3. Transformer Blocks: A stacked series of attention and feed-forward sub-layers.
    4. Language Modeling Head (lm_head): Projects the final representation vectors back into
       a probability distribution (logits) over our vocabulary to predict the next word.
       
    Weight Tying:
    We share the weights between the token embedding layer and the language modeling head.
    Since they perform complementary mappings (ID -> Vector and Vector -> Logits), sharing
    their parameters acts as a strong regularizer, drastically reduces total model size,
    and speeds up convergence.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        num_layers: int, 
        max_len: int = 128, 
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Total number of unique tokens in our vocabulary.
            d_model: Dimensionality of the token embedding space.
            num_heads: Number of attention heads in each Transformer Block.
            d_ff: Intermediate dimension in the Feed-Forward networks.
            num_layers: Number of Transformer Blocks to stack.
            max_len: Maximum sequence length supported by the positional encoding.
            dropout: Dropout probability.
        """
        super(TransformerChatbot, self).__init__()
        
        # 1. Token Embeddings
        # Maps integer token IDs to continuous vectors of size d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding (trigonometric sine/cosine vectors)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # 3. Stacked Transformer Blocks
        # Using nn.ModuleList allows PyTorch to register the list of custom modules correctly
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final normalization layer to stabilize activations before projection
        self.ln_final = nn.LayerNorm(d_model)
        
        # 5. Language modeling head
        # Projects the final token representation vectors (d_model) to vocabulary size probabilities
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: Share the weights of token_embeddings and lm_head
        # Crucial for deep NLP models
        self.token_embeddings.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of token IDs: shape (batch_size, seq_len)
            mask: Optional causal/look-ahead mask: shape (seq_len, seq_len)
        Returns:
            logits: Log-probabilities over the vocabulary for each token: shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Look up token embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)
        
        # Step 2: Add Positional Encodings
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        
        # Step 3: Pass sequentially through each Transformer Block
        for block in self.blocks:
            x = block(x, mask=mask)
            
        # Step 4: Apply final Layer Normalization
        x = self.ln_final(x)
        
        # Step 5: Project to vocabulary space to compute next-token logits
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 5B DEMO: FULL MODEL ARCHITECTURE ===")
    
    # 1. Setup Hyperparameters
    vocab_size = 50     # Size of our mock vocabulary
    d_model = 8         # Small embedding dim
    num_heads = 2
    d_ff = 32
    num_layers = 2      # Stack of 2 blocks
    max_len = 16
    
    print(f"\nConfiguration:")
    print(f"- Vocab Size: {vocab_size}")
    print(f"- Stacked Layers: {num_layers}")
    print(f"- Embedding Dimension: {d_model}")
    print(f"- Attention Heads: {num_heads}")
    
    # 2. Instantiate Model
    model = TransformerChatbot(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    )
    
    # 3. Simulate raw token inputs: (batch_size=2, seq_len=6)
    torch.manual_seed(42)
    mock_input_ids = torch.randint(low=0, high=vocab_size, size=(2, 6))
    print(f"\nInput IDs shape: {mock_input_ids.shape}")
    print(mock_input_ids)
    
    # 4. Create causal mask for sequence length of 6
    seq_len = mock_input_ids.size(1)
    mask_template = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = torch.zeros(seq_len, seq_len).masked_fill(mask_template == 0, float('-inf'))
    
    # 5. Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(mock_input_ids, mask=causal_mask)
        
    print(f"\nOutput Logits shape: {logits.shape}")
    
    # 6. Verify Shapes
    print("\n--- Verifying Model Shapes & Tie Weight ---")
    assert logits.shape == (2, 6, vocab_size), f"Expected shape {(2, 6, vocab_size)}, got {logits.shape}"
    assert id(model.token_embeddings.weight) == id(model.lm_head.weight), "Weight tying failed!"
    print("SUCCESS: Full Transformer chatbot model compiled and verified perfectly!")
