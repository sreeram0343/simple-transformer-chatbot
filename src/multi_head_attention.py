import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention from the paper "Attention Is All You Need".
    
    Why it exists:
    Rather than performing attention once over the full embedding space, Multi-Head Attention splits
    the embedding dimension (d_model) into 'num_heads' smaller spaces (of size d_k).
    Each head performs self-attention independently in parallel. This allows the model to simultaneously
    attend to information from different representation subspaces at different positions. 
    For example, one head might learn to connect adjectives with nouns, while another connects pronouns
    with their referenced nouns.
    
    Tensor Shape Walkthrough:
    - Input: (batch_size, seq_len, d_model)
    - Projected Q, K, V: (batch_size, seq_len, d_model)
    - Split into heads & transposed: (batch_size, num_heads, seq_len, d_k)
    - Attention matrix: (batch_size, num_heads, seq_len, seq_len)
    - Weighted values (output of attention): (batch_size, num_heads, seq_len, d_k)
    - Concatenated heads: (batch_size, seq_len, d_model)
    - Final linear output projection: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: The input embedding dimension (must be divisible by num_heads).
            num_heads: The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Ensure that the embedding dimension is perfectly divisible by the number of heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"Embedding dim d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
            
        # The dimension of each individual head's Query, Key, and Value space
        self.d_k = d_model // num_heads
        
        # Linear projection layers for Query, Key, and Value
        # We can project all heads in parallel using a single large weight matrix of size (d_model, d_model).
        # This is mathematically equivalent and much faster than having individual layers per head.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # The final output linear layer to project the concatenated head outputs back
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            mask: Optional tensor of shape (seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
                  used to mask future tokens during decoding.
        Returns:
            output: Contextualized output embeddings: shape (batch_size, seq_len, d_model)
            attention_weights: Normalized attention weights of the first head (for visualization):
                               shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project input into Q, K, and V
        # Output shape: (batch_size, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Step 2: Split the projected tensors into multiple heads
        # - Original shape: (batch_size, seq_len, d_model)
        # - Reshaped shape: (batch_size, seq_len, num_heads, d_k)
        # - Transposed shape: (batch_size, num_heads, seq_len, d_k)
        # Transposing swaps seq_len and num_heads. This lets PyTorch treat num_heads
        # as a batch dimension, allowing us to compute attention on all heads in parallel.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute raw attention scores (scaled dot product)
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K^T: (batch_size, num_heads, d_k, seq_len)
        # Scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Step 4: Apply causal masking (if provided)
        # Since mask is shape (seq_len, seq_len) or (batch_size, 1, seq_len, seq_len),
        # PyTorch will broadcast it to (batch_size, num_heads, seq_len, seq_len) automatically.
        if mask is not None:
            scores = scores + mask
            
        # Step 5: Softmax normalization to get attention probabilities
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 6: Multiply attention weights by Value vectors
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_k)
        # Output shape: (batch_size, num_heads, seq_len, d_k)
        head_outputs = torch.matmul(attention_weights, V)
        
        # Step 7: Concatenate all heads back together
        # - Transpose back: (batch_size, seq_len, num_heads, d_k)
        # - View/reshape: (batch_size, seq_len, d_model) (merges the heads together)
        # contiguous() is called to allocate a new contiguous chunk of memory after transpose,
        # which is required by PyTorch before calling view().
        concat_outputs = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Step 8: Apply the final linear output projection
        # Shape: (batch_size, seq_len, d_model)
        output = self.out_proj(concat_outputs)
        
        # We return the output and the attention weights of the first head for inspection
        return output, attention_weights[:, 0, :, :]


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 3 DEMO: MULTI-HEAD ATTENTION ===")
    
    # 1. Setup Hyperparameters
    batch_size = 1
    seq_len = 4
    d_model = 8         # Must be divisible by num_heads!
    num_heads = 2       # We will use 2 heads (each head dimension d_k = 8 / 2 = 4)
    
    print(f"\nConfiguration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length (seq_len): {seq_len}")
    print(f"- Embedding Dimension (d_model): {d_model}")
    print(f"- Number of Heads (h): {num_heads}")
    print(f"- Dimension per Head (d_k): {d_model // num_heads}")

    # 2. Instantiate Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 3. Create simulated input tensor
    torch.manual_seed(42)
    mock_input = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput Tensor Shape: {mock_input.shape}")

    # 4. Create causal mask (look-ahead mask)
    mask_template = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = torch.zeros(seq_len, seq_len).masked_fill(mask_template == 0, float('-inf'))
    
    # 5. Forward Pass
    output, first_head_weights = mha(mock_input, mask=causal_mask)
    
    print(f"\nOutput Shape: {output.shape}")
    print(f"First Head Attention Weights Shape: {first_head_weights.shape}")
    
    print("\nAttention Matrix for First Head:")
    print(first_head_weights[0])
    
    # Assertions to confirm correctness
    print("\n--- Verifying Output and Tensor Shapes ---")
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert first_head_weights.shape == (batch_size, seq_len, seq_len), f"Expected shape {(batch_size, seq_len, seq_len)}, got {first_head_weights.shape}"
    
    # Check that mask blocks future lookahead in the returned first-head weights
    assert first_head_weights[0][0][1].item() == 0.0
    assert first_head_weights[0][0][2].item() == 0.0
    assert first_head_weights[0][0][3].item() == 0.0
    
    print("SUCCESS: Multi-Head Attention shapes and masking behave exactly as designed!")
