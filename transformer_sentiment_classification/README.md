# Transformer-Based Sentiment Classification

A real-world use case implementation of the **Transformer** architecture
applied to **sentiment analysis** (text classification).  Every component of
the Transformer encoder is built from scratch with PyTorch so that each part
of the architecture can be studied directly in the code.

---

## Use Case

**Goal:** Given a short movie-review sentence, predict whether it expresses a
**positive** or **negative** sentiment.

This is a classic NLP classification task.  The Transformer encoder converts
the input token sequence into contextualised representations, and a linear
head maps the `[CLS]` token's representation to a class label.

---

## Transformer Architecture Components

The diagram below shows how components are connected:

```
Input tokens
     │
     ▼
┌─────────────────────┐
│   Token Embedding   │  Learnable d_model-dim vectors per token
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Positional Encoding │  Sinusoidal PE added to inject sequence order
└─────────────────────┘
     │
     ▼  (repeated N times)
┌────────────────────────────────────────────┐
│              Encoder Layer                 │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │     Multi-Head Self-Attention        │  │
│  │  Q, K, V ← same input (self-attn)   │  │
│  │  scores = softmax(QKᵀ / √d_k) · V  │  │
│  └──────────────────────────────────────┘  │
│       │   + residual connection            │
│  ┌────┴─────────┐                          │
│  │  LayerNorm   │                          │
│  └──────────────┘                          │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Position-wise Feed-Forward (FFN)   │  │
│  │  FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂   │  │
│  └──────────────────────────────────────┘  │
│       │   + residual connection            │
│  ┌────┴─────────┐                          │
│  │  LayerNorm   │                          │
│  └──────────────┘                          │
└────────────────────────────────────────────┘
     │
     ▼
  [CLS] token representation  (position 0)
     │
     ▼
┌──────────────────┐
│  Linear + softmax│  Maps d_model → num_classes
└──────────────────┘
     │
     ▼
  Sentiment label (POSITIVE / NEGATIVE)
```

### Component Details

| Component | File | Key idea |
|-----------|------|----------|
| **TokenEmbedding** | `transformer.py` | Converts discrete token IDs into continuous d_model-dimensional vectors. Embeddings are scaled by √d_model as in the original paper. |
| **PositionalEncoding** | `transformer.py` | Adds fixed sinusoidal signals to embeddings so the model can distinguish token positions. Uses `sin` at even dimensions and `cos` at odd dimensions. |
| **MultiHeadAttention** | `transformer.py` | Computes scaled dot-product attention in `h` parallel subspaces. Each head learns to attend to different aspects of the input. Supports a padding mask to ignore padding tokens. |
| **FeedForward** | `transformer.py` | Two linear projections with a ReLU activation between them, applied identically to each position. Widens the representation space (inner dim = 4× d_model by default). |
| **EncoderLayer** | `transformer.py` | One Transformer encoder block: MultiHeadAttention → Add & Norm → FFN → Add & Norm. |
| **TransformerEncoder** | `transformer.py` | Stacks N `EncoderLayer` modules sequentially. |
| **TransformerClassifier** | `transformer.py` | Full model: embeddings → positional encoding → N encoder layers → [CLS] pooling → linear classifier. |

---

## Project Structure

```
transformer_sentiment_classification/
├── transformer.py      # All Transformer components (core architecture)
├── dataset.py          # Tokenizer, SentimentDataset, DataLoader builder
├── train.py            # Training loop with logging and checkpointing
├── inference.py        # Inference demo (single sentence or interactive)
├── test_transformer.py # pytest unit tests for every component
└── requirements.txt    # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd transformer_sentiment_classification
python train.py
```

Sample output:

```
Using device: cpu

Vocabulary size : 147
Training samples: 24
Validation samples: 6

Model parameters: 201,218

 Epoch | Train Loss | Train Acc |  Val Loss | Val Acc |   Time
-----------------------------------------------------------------
     1 |     0.6938 |    0.5000 |    0.6945 |  0.5000 |   0.1s
    ...
    30 |     0.1823 |    1.0000 |    0.2541 |  1.0000 |   0.1s ✓

Best validation accuracy: 1.0000
Model checkpoint saved to best_model.pt
```

### 3. Run inference

```bash
# Classify a single sentence:
python inference.py --text "This film was absolutely fantastic!"

# Interactive mode:
python inference.py
```

Example output:

```
Input   : This film was absolutely fantastic!
Sentiment: POSITIVE 😊  (confidence: 97.3%)
```

### 4. Run the tests

```bash
pytest test_transformer.py -v
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Model dimensionality (d_model) |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 2 | Encoder depth (N) |
| `ff_dim` | 256 | FFN inner dimension |
| `max_len` | 64 | Maximum sequence length |
| `dropout` | 0.1 | Dropout probability |
| `lr` | 3e-4 | AdamW learning rate |
| `epochs` | 30 | Training epochs |

Adjust them via command-line flags:

```bash
python train.py --epochs 50 --embed-dim 256 --num-heads 8 --num-layers 4
```

---

## Self-Attention Deep Dive

The core formula of Scaled Dot-Product Attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Where:
- **Q** (query), **K** (key), **V** (value) are linear projections of the input.
- Dividing by √d_k prevents the dot products from growing too large in magnitude,
  which would push softmax into regions with very small gradients.
- Multi-head attention runs this computation `h` times in parallel subspaces, then
  concatenates and projects:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
  https://arxiv.org/abs/1706.03762
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*.
  https://arxiv.org/abs/1810.04805
