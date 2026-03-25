"""
Dataset utilities for Transformer sentiment classification.

Provides:
    - SimpleTokenizer: Character/word-level tokenizer with a built vocabulary.
    - SentimentDataset: PyTorch Dataset wrapping text/label pairs.
    - collate_fn: Pads batches of variable-length sequences.
    - get_dataloaders: Builds train/validation DataLoaders from sample data.

The sample dataset is a small collection of English movie-review sentences
labelled as positive (1) or negative (0). It is intentionally small so the
entire assignment runs without downloading external data.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DATA: list[tuple[str, int]] = [
    # Positive examples (label = 1)
    ("This movie was absolutely fantastic and I loved every moment of it", 1),
    ("An outstanding performance by all the actors in this wonderful film", 1),
    ("The storyline was engaging and kept me glued to my seat throughout", 1),
    ("Brilliant direction and beautiful cinematography made this a masterpiece", 1),
    ("I highly recommend this film to anyone who loves great storytelling", 1),
    ("The special effects were stunning and the plot was very well written", 1),
    ("Such an emotional and moving experience that I will never forget", 1),
    ("A feel good movie with excellent humor and great character development", 1),
    ("The soundtrack was wonderful and perfectly complemented the visuals", 1),
    ("One of the best films I have ever watched in my entire life", 1),
    ("Incredible acting and a deeply touching story that resonates with everyone", 1),
    ("I was thoroughly entertained from beginning to end what a great film", 1),
    ("The director did a phenomenal job bringing this story to life on screen", 1),
    ("Absolutely loved the chemistry between the lead actors in this movie", 1),
    ("A timeless classic that will be remembered for generations to come", 1),
    # Negative examples (label = 0)
    ("This movie was terrible and a complete waste of my time and money", 0),
    ("Awful script with poor acting and an incredibly boring plot overall", 0),
    ("I fell asleep halfway through because the movie was so painfully dull", 0),
    ("The worst film I have ever seen in my life do not watch this", 0),
    ("Dreadful storytelling with characters that are completely unbelievable", 0),
    ("The special effects looked cheap and the dialogue was really awful", 0),
    ("Disappointing in every way possible this movie had no redeeming qualities", 0),
    ("The plot made no sense and the ending was incredibly unsatisfying", 0),
    ("A total disaster of a film that should never have been made at all", 0),
    ("Poor direction and terrible pacing ruined what could have been decent", 0),
    ("I wanted to walk out of the theater after the first twenty minutes", 0),
    ("Clichéd and predictable with nothing original or interesting to offer", 0),
    ("The acting was wooden and the script was full of plot holes everywhere", 0),
    ("Completely forgettable and uninspired with a mediocre cast throughout", 0),
    ("An absolute bore from start to finish with zero entertainment value", 0),
]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}


class SimpleTokenizer:
    """Minimal whitespace tokenizer with a word-level vocabulary.

    Special tokens are added at indices 0–2:
        0 = <PAD>  (padding token, ignored in attention)
        1 = <UNK>  (out-of-vocabulary token)
        2 = <CLS>  (classification token prepended to every sequence)

    Args:
        texts: Iterable of raw strings to build the vocabulary from.
        min_freq: Minimum word frequency required to be kept in the vocabulary.
    """

    def __init__(self, texts: list[str], min_freq: int = 1) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        self.token2id: dict[str, int] = dict(SPECIAL_TOKENS)
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.token2id:
                self.token2id[word] = len(self.token2id)

        self.id2token: dict[int, str] = {v: k for k, v in self.token2id.items()}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lower-case and split on non-alphanumeric characters."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """Convert a text string to a list of token IDs.

        Prepends the <CLS> token and truncates to max_len (inclusive of CLS).

        Args:
            text: Raw input string.
            max_len: Maximum sequence length including the <CLS> token.

        Returns:
            List of token IDs, starting with the <CLS> ID.
        """
        tokens = self._tokenize(text)
        ids = [self.token2id.get(t, SPECIAL_TOKENS["<UNK>"]) for t in tokens]
        # Prepend CLS and truncate.
        ids = [SPECIAL_TOKENS["<CLS>"]] + ids[: max_len - 1]
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification.

    Args:
        data: List of (text, label) tuples.
        tokenizer: Fitted SimpleTokenizer instance.
        max_len: Maximum token sequence length.
    """

    def __init__(
        self,
        data: list[tuple[str, int]],
        tokenizer: SimpleTokenizer,
        max_len: int = 64,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text, label = self.data[idx]
        ids = self.tokenizer.encode(text, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences in a batch to the same length.

    Args:
        batch: List of (token_ids, label) pairs from SentimentDataset.

    Returns:
        Tuple of:
            - input_ids: (batch_size, max_seq_len) padded with zeros.
            - padding_mask: (batch_size, max_seq_len) bool tensor; True = pad.
            - labels: (batch_size,) class indices.
    """
    token_seqs, labels = zip(*batch)
    max_len = max(t.size(0) for t in token_seqs)

    padded = torch.zeros(len(token_seqs), max_len, dtype=torch.long)
    padding_mask = torch.ones(len(token_seqs), max_len, dtype=torch.bool)

    for i, seq in enumerate(token_seqs):
        length = seq.size(0)
        padded[i, :length] = seq
        padding_mask[i, :length] = False  # non-pad positions → False

    return padded, padding_mask, torch.stack(labels)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def get_dataloaders(
    batch_size: int = 8,
    max_len: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """Build train and validation DataLoaders from the built-in sample data.

    Args:
        batch_size: Number of samples per batch.
        max_len: Maximum sequence length (including <CLS> token).
        val_split: Fraction of data to use for validation.
        seed: Random seed for the split.

    Returns:
        Tuple of (train_loader, val_loader, tokenizer).
    """
    import random

    random.seed(seed)
    data = list(SAMPLE_DATA)
    random.shuffle(data)

    split = int(len(data) * (1 - val_split))
    train_data = data[:split]
    val_data = data[split:]

    # Fit tokenizer on training texts only (to avoid data leakage).
    tokenizer = SimpleTokenizer([text for text, _ in train_data])

    train_ds = SentimentDataset(train_data, tokenizer, max_len)
    val_ds = SentimentDataset(val_data, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, tokenizer
