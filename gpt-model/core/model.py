# This is the model.py file for the GPTlite model
# Author: Chloe Lavrat
# Date: 2024-12-11

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Head(nn.Module):
    """Single attention head that performs scaled dot-product attention with causal masking."""

    def __init__(self, head_size: int, embedding_dim: int, dropout: float, block_size: int):
        super().__init__()
        # Linear projections for key, query, and value
        self.key_projection = nn.Linear(embedding_dim, head_size, bias=False)
        self.query_projection = nn.Linear(embedding_dim, head_size, bias=False)
        self.value_projection = nn.Linear(embedding_dim, head_size, bias=False)

        # Causal mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the attention head."""
        batch_size, sequence_length, embedding_dim = x.shape

        # Project input into key, query, and value vectors
        keys = self.key_projection(x)
        queries = self.query_projection(x)
        values = self.value_projection(x)

        # Compute attention scores
        attention_scores = queries @ keys.transpose(-2, -1)

        # Scale attention scores by sqrt(d_k) so that variance of the weights is 1
        attention_scores *= embedding_dim ** -0.5

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(
            self.tril[:sequence_length, :sequence_length] == 0,
            float('-inf')
        )

        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute output
        output = attention_weights @ values
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism that combines multiple attention heads in parallel."""

    def __init__(self, num_attention_heads: int, head_dimension: int, embedding_dimension: int, dropout_rate: float, block_size: int):
        super().__init__()
        # Create multiple attention heads in parallel
        self.attention_heads = nn.ModuleList([
            Head(head_dimension, embedding_dimension, dropout_rate, block_size)
            for _ in range(num_attention_heads)
        ])

        # Project concatenated attention outputs back to embedding dimension
        self.output_projection = nn.Linear(
            head_dimension * num_attention_heads, embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through all attention heads."""
        # Process input through all attention heads in parallel
        attention_outputs = [head(embeddings) for head in self.attention_heads]

        # Concatenate outputs from all heads
        concatenated = torch.cat(attention_outputs, dim=-1)

        # Project back to embedding dimension and apply dropout
        output = self.output_projection(concatenated)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network that expands input dimension, applies non-linearity,
    and projects back to original dimension."""

    def __init__(self, dim: int, dropout: float, expansion_factor: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim),  # Expand dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * dim, dim),  # Project back
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block combining multi-head self-attention and feed-forward layers with residual connections."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float, block_size: int):
        super().__init__()
        # Calculate attention head size
        attention_head_size = embedding_dim // num_heads

        # Multi-head self-attention layer
        self.attention = MultiHeadAttention(
            num_attention_heads=num_heads,
            head_dimension=attention_head_size,
            embedding_dimension=embedding_dim,
            dropout_rate=dropout_rate,
            block_size=block_size
        )

        # Position-wise feed-forward layer
        self.feed_forward = FeedForward(
            dim=embedding_dim, dropout=dropout_rate)

        # Layer normalization
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.pre_feedforward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward layers with residual connections."""
        # Self-attention with residual connection
        attention_output = hidden_states + \
            self.attention(self.pre_attention_norm(hidden_states))

        # Feed-forward with residual connection
        output = attention_output + \
            self.feed_forward(self.pre_feedforward_norm(attention_output))

        return output


class GPTlite(nn.Module):
    """A lightweight GPT-style transformer model for language modeling and generation."""

    def __init__(self, config: dict):
        super().__init__()
        # Model configuration
        self.context_size = config['context_size']
        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        dropout = config['dropout']

        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(
            self.context_size, embedding_dim)

        # Transformer blocks with linearly increasing dropout
        self.blocks = nn.Sequential(*[
            Block(embedding_dim, num_heads,
                  dropout, self.context_size)
            for i in range(num_layers)
        ])

        # Layer scaling for better gradient flow
        self.layer_scales = nn.Parameter(torch.ones(num_layers) * 0.1)

        # Output layers
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model."""
        batch_size, seq_length = input_tokens.shape

        # Get embeddings
        token_emb = self.token_embeddings(input_tokens)
        pos_emb = self.position_embeddings(
            torch.arange(seq_length, device=input_tokens.device))
        hidden_states = token_emb + pos_emb

        # Process through transformer blocks
        hidden_states = self.blocks(hidden_states)

        # Generate output logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_projection(hidden_states)

        # Compute loss if targets provided
        if targets is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
        else:
            loss = None

        return logits, loss

    def generate(self, initial_tokens: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: Optional[int] = 40) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        tokens = initial_tokens.clone()

        for _ in range(max_new_tokens):
            # Get predictions for next token
            context = tokens[:, -self.context_size:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k is not None:
                threshold_value, _ = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits <
                                  threshold_value[:, [-1]]] = float('-inf')

            # Sample next token
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens
