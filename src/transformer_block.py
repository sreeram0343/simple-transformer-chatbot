import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    Implements a single GPT-style Decoder-only Transformer Block.
    
    Why it exists:
    A Transformer Block is the fundamental building block of modern generative models (like GPT).
    Deep Transformer models are created by stacking multiple copies of this block on top of each other.
    Each block processes the sequence, extracting higher-level representations.
    
    It consists of two main sub-layers:
    1. A Multi-Head Self-Attention layer (with Causal Masking).
    2. A Position-wise Feed-Forward Network (FFN).
    
    To ensure stable training in deep networks, we wrap each sub-layer with:
    - Residual Connections: Adds the input directly to the output (x + SubLayer(x)).
    - Layer Normalization (Pre-LN style): Normalizes the inputs *before* they enter the sub-layers.
    
    Architecture Flow (Pre-LN):
        Input x -> LayerNorm1 -> MultiHeadAttention -> Add input x (Residual Connection) -> Temp_x
        Temp_x -> LayerNorm2 -> FeedForward -> Add Temp_x (Residual Connection) -> Output
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: The embedding dimension of the tokens.
            num_heads: The number of attention heads.
            d_ff: The dimension of the inner hidden layer of the Feed-Forward Network.
            dropout: The dropout rate for regularization.
        """
        super(TransformerBlock, self).__init__()
        
        # 1. Multi-Head Attention Sub-layer
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # 2. Position-wise Feed-Forward Network Sub-layer
        # Consists of: Linear projection to d_ff -> Activation -> Dropout -> Linear projection back to d_model
        # The paper uses a expansion factor of 4 (e.g. if d_model=512, d_ff=2048)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Modern models (like GPT-3/BERT) use GELU (Gaussian Error Linear Unit) instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Layer Normalization modules
        # LayerNorm normalizes across the embedding dimension (d_model) for each token independently
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # 4. Dropout layers for regularization
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional causal mask tensor of shape (seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # --- Sub-layer 1: Multi-Head Attention (with Pre-LN) ---
        # 1. Apply LayerNorm first (Pre-LN style)
        normed_x1 = self.ln_1(x)
        # 2. Pass normalized tensor through Multi-Head Attention
        attention_output, _ = self.mha(normed_x1, mask=mask)
        # 3. Apply Dropout for regularization and add the original input x (Residual Connection)
        x = x + self.dropout_1(attention_output)
        
        # --- Sub-layer 2: Feed-Forward Network (with Pre-LN) ---
        # 1. Apply LayerNorm first
        normed_x2 = self.ln_2(x)
        # 2. Pass through the Feed-Forward Network
        ffn_output = self.ffn(normed_x2)
        # 3. Add to the input x (Residual Connection)
        x = x + ffn_output
        
        return x


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 4 DEMO: TRANSFORMER BLOCK ===")
    
    # 1. Setup Hyperparameters
    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32           # FFN hidden layer dimension (often 4x d_model)
    
    print(f"\nConfiguration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length (seq_len): {seq_len}")
    print(f"- Embedding Dimension (d_model): {d_model}")
    print(f"- Attention Heads: {num_heads}")
    print(f"- FFN Hidden Dimension (d_ff): {d_ff}")

    # 2. Instantiate the Transformer Block
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1)
    
    # 3. Create simulated input tensor (representing token embeddings + positional encodings)
    torch.manual_seed(42)
    mock_input = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput Tensor Shape: {mock_input.shape}")

    # 4. Create causal mask
    mask_template = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = torch.zeros(seq_len, seq_len).masked_fill(mask_template == 0, float('-inf'))
    
    # 5. Forward Pass
    # Turn off training mode (evaluation mode) to disable dropout for consistent evaluation shapes
    block.eval()
    with torch.no_grad():
        output = block(mock_input, mask=causal_mask)
        
    print(f"\nOutput Shape: {output.shape}")
    print("\nOutput Tensor:")
    print(output)

    # 6. Verify Shapes
    print("\n--- Verifying Shapes ---")
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("SUCCESS: Transformer Block forward pass ran correctly, output maintains exact shape!")
