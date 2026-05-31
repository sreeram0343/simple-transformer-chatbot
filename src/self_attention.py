import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class SelfAttention(nn.Module):
    """
    Implements a single-head Scaled Dot-Product Self-Attention mechanism with support for Causal Masking.
    
    Why it exists:
    Self-Attention allows every word in a sequence to look at ("attend to") every other word in the sequence,
    determining how much weight to place on other words to contextualize its own meaning.
    
    The Query-Key-Value (QKV) Analogy:
    - Query (Q): "What am I looking for?" (Represented by the current word)
    - Key (K): "What information do I contain?" (Represented by all words in the sequence)
    - Value (V): "What actual content/meaning do I carry?" (The content to extract once we find a match)
    
    Mathematical Formula:
        Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V
    """
    def __init__(self, d_model: int, d_k: int) -> None:
        """
        Args:
            d_model (int): The embedding dimension of the input sequence.
            d_k (int): The projection dimension for Queries and Keys.
        """
        super(SelfAttention, self).__init__()
        self.d_k: int = d_k
        
        # Learnable weight projections for Query, Key, and Value
        # These project the input embedding dimension (d_model) into the attention subspace (d_k)
        self.q_proj: nn.Linear = nn.Linear(d_model, d_k, bias=False)
        self.k_proj: nn.Linear = nn.Linear(d_model, d_k, bias=False)
        self.v_proj: nn.Linear = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): Optional tensor of shape (seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
                  used to block attention to future tokens (causal/look-ahead mask).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - output: Attention-contextualized embeddings: shape (batch_size, seq_len, d_k)
                - attention_weights: Normalized attention scores: shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Project input X into Query, Key, and Value tensors
        # Q, K, V shapes: (batch_size, seq_len, d_k)
        Q: torch.Tensor = self.q_proj(x)
        K: torch.Tensor = self.k_proj(x)
        V: torch.Tensor = self.v_proj(x)
        
        # Step 2: Compute raw attention scores (dot product between Query and Key)
        # Matrix multiplication: (batch_size, seq_len, d_k) x (batch_size, d_k, seq_len) -> (batch_size, seq_len, seq_len)
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 3: Scale the scores by sqrt(d_k) to prevent extremely large values
        scores = scores / math.sqrt(self.d_k)
        
        # Step 4: Apply the Causal Mask (if provided)
        if mask is not None:
            scores = scores + mask
            
        # Step 5: Normalize attention scores along the last dimension
        attention_weights: torch.Tensor = F.softmax(scores, dim=-1)
        
        # Step 6: Multiply the weights by the Values (weighted sum of content)
        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, d_k) -> (batch_size, seq_len, d_k)
        output: torch.Tensor = torch.matmul(attention_weights, V)
        
        return output, attention_weights


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 2 DEMO: SELF-ATTENTION ===")
    
    # 1. Setup Hyperparameters
    batch_size = 1
    seq_len = 4         # e.g., "I", "love", "ai", "bots"
    d_model = 8         # Embedding dimension
    d_k = 8             # Projection dimension

    print(f"\nConfiguration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length (seq_len): {seq_len}")
    print(f"- Input Embedding Dim (d_model): {d_model}")
    print(f"- Attention Project Dim (d_k): {d_k}")

    # 2. Instantiate Self-Attention
    attention_layer = SelfAttention(d_model=d_model, d_k=d_k)

    # 3. Simulate input tensor (output from embedding + positional encoding)
    # We will use random numbers but with seed for reproducibility
    torch.manual_seed(42)
    mock_input = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput Tensor Shape: {mock_input.shape}")

    # 4. Create a Causal Mask (Look-ahead Mask)
    # A chatbot is autoregressive: position 0 can only look at 0.
    # Position 1 can look at 0, 1.
    # Position 2 can look at 0, 1, 2.
    # We create a lower-triangular matrix of ones, and then invert it to get upper-triangular zeros.
    # 1s on/below diagonal, 0s above diagonal:
    mask_template = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\nLower Triangular Mask Template (1 = allowed, 0 = blocked):")
    print(mask_template)
    
    # We map 1 -> 0.0 (no change in score) and 0 -> -1e9 (reduces attention probability to 0 in softmax)
    causal_mask = torch.zeros(seq_len, seq_len)
    causal_mask = causal_mask.masked_fill(mask_template == 0, float('-inf'))
    print(f"\nCausal Mask values (0.0 = allowed, -inf = blocked):")
    print(causal_mask)

    # 5. Forward Pass WITHOUT Mask (Bidirectional Attention, like BERT)
    output_no_mask, weights_no_mask = attention_layer(mock_input, mask=None)
    
    print("\n--- Bidirectional Attention (No Mask) ---")
    print(f"Output Shape: {output_no_mask.shape}")
    print(f"Attention Weights Shape: {weights_no_mask.shape}")
    print("Attention Weights Matrix (rows are queries, cols are keys):")
    print(weights_no_mask[0]) # Print first batch element
    # Rows should sum to 1.0
    print(f"Rows sum: {weights_no_mask[0].sum(dim=-1)}")

    # 6. Forward Pass WITH Causal Mask (Autoregressive Attention, like GPT)
    output_masked, weights_masked = attention_layer(mock_input, mask=causal_mask)
    
    print("\n--- Autoregressive Attention (With Causal Mask) ---")
    print(f"Output Shape: {output_masked.shape}")
    print(f"Attention Weights Shape: {weights_masked.shape}")
    print("Causal Attention Weights Matrix (upper triangular must be EXACTLY 0.0):")
    print(weights_masked[0])
    print(f"Rows sum: {weights_masked[0].sum(dim=-1)}")

    # Math Verification of causal mask behavior:
    # Row 0 (query 0) can only look at Key 0. So key 0 weight should be 1.0, and keys 1,2,3 should be 0.0.
    # Row 1 (query 1) can only look at Keys 0, 1. Keys 2, 3 must be 0.0.
    print("\n--- Mask Correctness Verification ---")
    assert torch.allclose(weights_masked[0][0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)
    assert weights_masked[0][1][2].item() == 0.0
    assert weights_masked[0][1][3].item() == 0.0
    print("SUCCESS: Causal masking is fully functioning and blocking future look-aheads!")
