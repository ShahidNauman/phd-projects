"""
Transformer Architecture Components for Text Classification.

This module implements the core building blocks of the Transformer architecture
as introduced in "Attention Is All You Need" (Vaswani et al., 2017), adapted
for a text classification (sentiment analysis) use case.

Components implemented:
    - TokenEmbedding: Maps token IDs to dense vectors.
    - PositionalEncoding: Injects position information using sinusoidal functions.
    - MultiHeadAttention: Scaled dot-product attention with multiple heads.
    - FeedForward: Position-wise fully-connected feed-forward network.
    - EncoderLayer: One Transformer encoder block (attention + FFN + residuals).
    - TransformerEncoder: Stack of encoder layers.
    - TransformerClassifier: Full model with encoder + classification head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Learnable embedding layer that maps token IDs to dense vectors.

    Args:
        vocab_size: Total number of unique tokens in the vocabulary.
        embed_dim: Dimensionality of the embedding space (d_model).
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) as in the original paper.
        return self.embedding(tokens) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to token embeddings.

    Allows the model to use information about the relative or absolute
    position of tokens in the sequence, since self-attention is
    position-agnostic by itself.

    Args:
        embed_dim: Dimensionality of the model (d_model).
        max_len: Maximum sequence length to pre-compute encodings for.
        dropout: Dropout probability applied after adding positional encoding.
    """

    def __init__(
        self, embed_dim: int, max_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build (max_len, embed_dim) matrix of sinusoidal values.
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )  # (embed_dim/2,)
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        # Register as buffer so it moves with the model (e.g., .to(device)) but
        # is not a learned parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tensor of the same shape with positional information added.
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Self-Attention.

    Splits the embedding dimension into `num_heads` heads, computes
    attention independently in each subspace, then concatenates and
    projects the results.

    Args:
        embed_dim: Total embedding dimension (d_model). Must be divisible by num_heads.
        num_heads: Number of parallel attention heads.
        dropout: Dropout applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k = d_v = d_model / h

        # Unified projection for queries, keys, values (more efficient than three separate ones).
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            key_padding_mask: Boolean mask of shape (batch_size, seq_len) where
                True indicates padding tokens that should be ignored.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        B, T, C = x.shape

        # Project input to Q, K, V and split into heads.
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)  # each: (B, T, C)

        # Reshape to (B, num_heads, T, head_dim).
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention.
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, h, T, T)

        if key_padding_mask is not None:
            # Expand mask to (B, 1, 1, T) so it broadcasts over heads and query positions.
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values.
        context = torch.matmul(attn_weights, v)  # (B, h, T, head_dim)

        # Concatenate heads and project back to embed_dim.
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN).

    Applied identically to each position:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    The inner dimension is typically 4x embed_dim (as in the original paper).

    Args:
        embed_dim: Input/output dimensionality.
        ff_dim: Inner (hidden) dimensionality of the FFN.
        dropout: Dropout applied between the two linear layers.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """A single Transformer Encoder layer.

    Each layer consists of:
        1. Multi-Head Self-Attention with residual connection and layer norm.
        2. Position-wise Feed-Forward Network with residual connection and layer norm.

    This follows the Post-LN formulation from the original paper:
        x = LayerNorm(x + Sublayer(x))

    Args:
        embed_dim: Model dimensionality (d_model).
        num_heads: Number of attention heads.
        ff_dim: FFN inner dimensionality.
        dropout: Dropout probability used throughout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Sub-layer 1: Self-Attention
        x = self.norm1(x + self.dropout(self.self_attn(x, key_padding_mask)))
        # Sub-layer 2: Feed-Forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N identical Transformer Encoder layers.

    Args:
        num_layers: Number of encoder layers (N).
        embed_dim: Model dimensionality.
        num_heads: Attention heads per layer.
        ff_dim: FFN inner dimensionality.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class TransformerClassifier(nn.Module):
    """Full Transformer-based text classifier.

    Architecture:
        Token Embedding → Positional Encoding → Transformer Encoder
        → [CLS] token pooling → Linear classifier

    The [CLS] token (position 0) representation after the encoder is used
    as the sequence-level representation for classification, following the
    BERT-style convention.

    Args:
        vocab_size: Vocabulary size including special tokens.
        num_classes: Number of output classes.
        embed_dim: Model dimensionality (d_model). Default: 128.
        num_heads: Number of attention heads. Default: 4.
        num_layers: Number of encoder layers. Default: 2.
        ff_dim: FFN inner dimensionality. Default: 256.
        max_len: Maximum sequence length. Default: 128.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        self.encoder = TransformerEncoder(
            num_layers, embed_dim, num_heads, ff_dim, dropout
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len).
            padding_mask: Boolean mask of shape (batch_size, seq_len) where
                True marks padding positions to ignore in attention.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.token_embedding(input_ids)  # (B, T, embed_dim)
        x = self.pos_encoding(x)  # (B, T, embed_dim)
        x = self.encoder(x, padding_mask)  # (B, T, embed_dim)

        # Use the [CLS] token (first position) as the sequence representation.
        cls_repr = x[:, 0, :]  # (B, embed_dim)
        cls_repr = self.dropout(cls_repr)
        return self.classifier(cls_repr)  # (B, num_classes)
