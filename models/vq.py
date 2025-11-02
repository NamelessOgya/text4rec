import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer Layer (VQ-VAE)
    
    Args:
        num_codes (int): Number of vectors in the codebook.
        embedding_dim (int): Dimensionality of the embedding vectors.
        commitment_cost (float): Weight for the commitment loss.
    """
    def __init__(self, num_codes, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        # Initialize weights as in the original VQ-VAE paper
        self.embedding.weight.data.uniform_(-1./self.num_codes, 1./self.num_codes)

    def forward(self, z):
        """
        Forward pass
        
        Args:
            z (torch.Tensor): Input tensor from the encoder. 
                              Shape: (Batch, SeqLen, Dim)
        
        Returns:
            dict: A dictionary containing:
                  - quantized (torch.Tensor): The quantized output tensor.
                  - loss (torch.Tensor): The VQ loss.
                  - indices (torch.Tensor): The indices of the closest codes.
        """
        # Reshape z to (Batch * SeqLen, Dim)
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # Calculate distances between input vectors and codebook vectors
        # distances from z to embeddings e_j: (z - e_j)^2 = z^2 - 2ze_j + e_j^2
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())
            
        # Find the closest codebook vector
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Get the quantized vectors
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # --- Calculate VQ Loss ---
        # 1. Codebook Loss: Move the codebook vectors towards the encoder outputs
        codebook_loss = F.mse_loss(z_q, z.detach())
        # 2. Commitment Loss: Encourage the encoder outputs to be close to the codebook vectors
        commitment_loss = F.mse_loss(z, z_q.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # --- Straight-Through Estimator ---
        # In the backward pass, the gradient will be copied from z_q to z
        quantized = z + (z_q - z).detach()
        
        return {
            'quantized': quantized,
            'loss': vq_loss,
            'indices': min_encoding_indices.view(z.shape[:-1])
        }
