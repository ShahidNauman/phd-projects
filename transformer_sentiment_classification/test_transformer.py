"""
Unit tests for the Transformer sentiment classification implementation.

Tests cover:
    - Individual architecture components (shapes, masks).
    - End-to-end forward pass.
    - Dataset and tokenizer utilities.
    - Collate function.

Run with:
    pytest test_transformer.py -v
"""

import math

import pytest
import torch

from dataset import (
    SimpleTokenizer,
    SentimentDataset,
    SAMPLE_DATA,
    collate_fn,
    get_dataloaders,
)
from transformer import (
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    TransformerEncoder,
    TransformerClassifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
EMBED_DIM = 32
NUM_HEADS = 4
FF_DIM = 64
NUM_LAYERS = 2
BATCH = 4
SEQ_LEN = 16


@pytest.fixture()
def tokenizer() -> SimpleTokenizer:
    texts = [text for text, _ in SAMPLE_DATA]
    return SimpleTokenizer(texts)


# ---------------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------------


class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(VOCAB_SIZE, EMBED_DIM)
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = emb(tokens)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_scaled_by_sqrt_d(self):
        """Embeddings should be scaled by sqrt(embed_dim)."""
        torch.manual_seed(0)
        emb = TokenEmbedding(VOCAB_SIZE, EMBED_DIM)
        tokens = torch.tensor([[1]])
        out = emb(tokens)
        raw = emb.embedding(tokens)
        assert torch.allclose(out, raw * math.sqrt(EMBED_DIM))


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(EMBED_DIM, max_len=64)
        x = torch.zeros(BATCH, SEQ_LEN, EMBED_DIM)
        out = pe(x)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_different_positions_differ(self):
        """Two different positions should have different positional codes."""
        pe = PositionalEncoding(EMBED_DIM, max_len=64, dropout=0.0)
        x = torch.zeros(1, 10, EMBED_DIM)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(EMBED_DIM, NUM_HEADS)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        out = attn(x)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_padding_mask_ignored(self):
        """Padding positions should not affect non-padding outputs."""
        torch.manual_seed(1)
        attn = MultiHeadAttention(EMBED_DIM, NUM_HEADS, dropout=0.0)
        attn.eval()

        x = torch.randn(2, SEQ_LEN, EMBED_DIM)
        # Mask the last 4 tokens of the second sample.
        mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
        mask[1, -4:] = True

        out_no_mask = attn(x)
        out_with_mask = attn(x, mask)

        # The first sample (no masked tokens) should be unaffected.
        assert not torch.allclose(out_no_mask[1], out_with_mask[1])

    def test_raises_on_bad_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=33, num_heads=4)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(EMBED_DIM, FF_DIM)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        assert ffn(x).shape == (BATCH, SEQ_LEN, EMBED_DIM)


# ---------------------------------------------------------------------------
# EncoderLayer
# ---------------------------------------------------------------------------


class TestEncoderLayer:
    def test_output_shape(self):
        layer = EncoderLayer(EMBED_DIM, NUM_HEADS, FF_DIM)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        assert layer(x).shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_with_mask(self):
        layer = EncoderLayer(EMBED_DIM, NUM_HEADS, FF_DIM)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        mask[:, -2:] = True
        out = layer(x, mask)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------


class TestTransformerEncoder:
    def test_output_shape(self):
        enc = TransformerEncoder(NUM_LAYERS, EMBED_DIM, NUM_HEADS, FF_DIM)
        x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        assert enc(x).shape == (BATCH, SEQ_LEN, EMBED_DIM)


# ---------------------------------------------------------------------------
# TransformerClassifier (end-to-end)
# ---------------------------------------------------------------------------


class TestTransformerClassifier:
    def test_output_shape(self):
        model = TransformerClassifier(
            vocab_size=VOCAB_SIZE,
            num_classes=2,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
        )
        input_ids = torch.randint(1, VOCAB_SIZE, (BATCH, SEQ_LEN))
        logits = model(input_ids)
        assert logits.shape == (BATCH, 2)

    def test_with_padding_mask(self):
        model = TransformerClassifier(
            vocab_size=VOCAB_SIZE,
            num_classes=3,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
        )
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        mask = input_ids == 0
        logits = model(input_ids, mask)
        assert logits.shape == (BATCH, 3)

    def test_different_inputs_give_different_outputs(self):
        torch.manual_seed(7)
        model = TransformerClassifier(
            vocab_size=VOCAB_SIZE,
            num_classes=2,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
            dropout=0.0,
        )
        model.eval()
        a = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN))
        b = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN))
        assert not torch.allclose(model(a), model(b))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestSimpleTokenizer:
    def test_vocab_contains_special_tokens(self, tokenizer):
        assert "<PAD>" in tokenizer.token2id
        assert "<UNK>" in tokenizer.token2id
        assert "<CLS>" in tokenizer.token2id

    def test_encode_starts_with_cls(self, tokenizer):
        ids = tokenizer.encode("great movie")
        assert ids[0] == tokenizer.token2id["<CLS>"]

    def test_encode_respects_max_len(self, tokenizer):
        ids = tokenizer.encode("a " * 100, max_len=10)
        assert len(ids) <= 10

    def test_unknown_word_maps_to_unk(self, tokenizer):
        ids = tokenizer.encode("xyznonexistentword123")
        # All word tokens should be UNK (index 1).
        unk_id = tokenizer.token2id["<UNK>"]
        # Position 0 is CLS, position 1 onwards are word tokens.
        assert all(i == unk_id for i in ids[1:])

    def test_vocab_size_matches_dict(self, tokenizer):
        assert tokenizer.vocab_size == len(tokenizer.token2id)


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------


class TestSentimentDataset:
    def test_length(self, tokenizer):
        ds = SentimentDataset(SAMPLE_DATA, tokenizer)
        assert len(ds) == len(SAMPLE_DATA)

    def test_item_types(self, tokenizer):
        ds = SentimentDataset(SAMPLE_DATA, tokenizer)
        ids, label = ds[0]
        assert isinstance(ids, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert ids.dtype == torch.long
        assert label.dtype == torch.long

    def test_label_in_range(self, tokenizer):
        ds = SentimentDataset(SAMPLE_DATA, tokenizer)
        for _, label in ds:
            assert label.item() in {0, 1}


class TestCollate:
    def test_padding_shape(self, tokenizer):
        ds = SentimentDataset(SAMPLE_DATA[:4], tokenizer)
        batch = [ds[i] for i in range(4)]
        input_ids, mask, labels = collate_fn(batch)
        # All sequences padded to the same length.
        assert input_ids.shape[0] == 4
        assert mask.shape == input_ids.shape
        assert labels.shape == (4,)

    def test_padding_positions_are_masked(self, tokenizer):
        ds = SentimentDataset(SAMPLE_DATA[:4], tokenizer, max_len=64)
        batch = [ds[i] for i in range(4)]
        input_ids, mask, _ = collate_fn(batch)
        # Where input_ids == 0 (padding), mask should be True.
        pad_positions = input_ids == 0
        assert mask[pad_positions].all()


class TestGetDataloaders:
    def test_returns_three_items(self):
        train_loader, val_loader, tokenizer = get_dataloaders(batch_size=4)
        assert train_loader is not None
        assert val_loader is not None
        assert tokenizer is not None

    def test_batch_shape(self):
        train_loader, _, _ = get_dataloaders(batch_size=4)
        input_ids, mask, labels = next(iter(train_loader))
        assert input_ids.dim() == 2
        assert mask.shape == input_ids.shape
        assert labels.dim() == 1
