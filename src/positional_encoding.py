import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the Sinusoidal Positional Encoding from the paper "Attention Is All You Need".
    
    Why it exists:
    Since Transformers process the entire input sequence in parallel, they have no inherent
    sense of word order. Positional encoding adds a unique, position-dependent signal 
    to each token's embedding vector so the model can learn relative and absolute word order.
    
    Trigonometric Formulation:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: The dimensionality of the embedding vectors (must be an even number).
            max_len: The maximum sequence length supported by the positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        
        # Ensure embedding dimension is even to handle matching sin/cos pairs neatly
        if d_model % 2 != 0:
            raise ValueError(f"Embedding dimension d_model must be even, got {d_model}")
            
        # Create a matrix of shape (max_len, d_model) filled with zeros to store encoding matrix
        pe = torch.zeros(max_len, d_model)
        
        # Create a tensor of positions representing pos in the equations: shape (max_len, 1)
        # e.g., [[0], [1], [2], ..., [max_len - 1]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term: 10000^(2i / d_model)
        # We do this in log space for numerical stability:
        # div_term = exp(2i * -log(10000) / d_model)
        # We take steps of 2 because each term is shared by both a sin and cos component.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Populate the positional encoding matrix:
        # Fill even indices (0, 2, 4, ...) with sine waves
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Fill odd indices (1, 3, 5, ...) with cosine waves
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        # This makes it easy to add it directly to our input batch via broadcasting
        pe = pe.unsqueeze(0)
        
        # register_buffer stores 'pe' in the module's state, but makes sure PyTorch knows
        # it is NOT a learnable parameter (it won't be updated by gradients during training).
        # It will also be moved to the correct device (CPU/GPU) automatically when calling .to(device).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input word embeddings of shape (batch_size, seq_len, d_model)
        Returns:
            Embeddings with positional encodings added: shape (batch_size, seq_len, d_model)
        """
        # x.size(1) gets the actual sequence length of the current batch (which might be < max_len)
        # We slice self.pe up to seq_len: self.pe[:, :x.size(1)] -> shape (1, seq_len, d_model)
        # Adding x + pe uses PyTorch broadcasting to replicate the position vectors for all items in the batch
        x = x + self.pe[:, :x.size(1)]
        return x


# =====================================================================
# DEMO / VERIFICATION ZONE
# =====================================================================
if __name__ == "__main__":
    print("=== STAGE 1 DEMO: POSITIONAL ENCODING ===")
    
    # 1. Hyperparameters
    batch_size = 2
    seq_len = 5         # Mock sentence length (e.g. "hello how are you today")
    embedding_dim = 6   # Keep dimension small (6) for readable output prints (must be even)
    
    print(f"\nConfiguration:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Sequence Length (seq_len): {seq_len}")
    print(f"- Embedding Dimension (d_model): {embedding_dim}")

    # 2. Instantiate the PositionalEncoding module
    pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=10)
    
    # 3. Simulate input word embeddings (mock output of nn.Embedding)
    # Filling with 0.0 makes it easy to see exactly what the Positional Encodings look like
    mock_embeddings = torch.zeros(batch_size, seq_len, embedding_dim)
    print(f"\nCreated mock input embeddings (shape: {mock_embeddings.shape}) filled with zeros:")
    print(mock_embeddings)

    # 4. Run the forward pass
    encoded_output = pos_encoder(mock_embeddings)
    
    print(f"\nOutput embeddings after Positional Encoding (shape: {encoded_output.shape}):")
    print(encoded_output)
    
    # Let's inspect the values of the positional encoding for the first element in the batch
    pe_values = pos_encoder.pe[:, :seq_len]
    print(f"\nExtracted PE buffer slice [1, seq_len, d_model] (shape: {pe_values.shape}):")
    print(pe_values)

    # 5. Math verification
    # For token at pos=1, dim=0 (even): sin(1 / 10000^(0/6)) = sin(1) = 0.8415
    # For token at pos=1, dim=1 (odd):  cos(1 / 10000^(0/6)) = cos(1) = 0.5403
    expected_sin_pos1_d0 = math.sin(1.0)
    expected_cos_pos1_d1 = math.cos(1.0)
    
    actual_sin = pe_values[0, 1, 0].item()
    actual_cos = pe_values[0, 1, 1].item()
    
    print("\n--- Math Verification for Token at Position 1 ---")
    print(f"Index 0 (sin): Expected = {expected_sin_pos1_d0:.4f}, Actual = {actual_sin:.4f}")
    print(f"Index 1 (cos): Expected = {expected_cos_pos1_d1:.4f}, Actual = {actual_cos:.4f}")
    
    assert abs(actual_sin - expected_sin_pos1_d0) < 1e-5
    assert abs(actual_cos - expected_cos_pos1_d1) < 1e-5
    print("SUCCESS: Trigonometric values match the theory perfectly!")
