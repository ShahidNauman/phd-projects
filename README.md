# IST Assignments

## Transformer Assignment (Beginner-Friendly)

This repository now includes a simple and practical Transformer use case:

- **Use case:** Real-world **customer feedback text classification**
- **Task type:** Sentiment classification (POSITIVE / NEGATIVE)
- **Architecture used:** Pre-trained **DistilBERT Transformer** via Hugging Face pipeline
- **File:** `transformer_text_classification_demo.py`

This is one of the easiest ways to understand practical Transformer usage without building a model from scratch.

---

## Why this is a good starter project

1. Minimal code complexity.
2. Real-world value (feedback monitoring/customer support triage).
3. Includes **real-time classification** mode (type text and get instant prediction).
4. Code has extensive comments for Python beginners.

---

## Setup

Install dependencies:

```bash
pip install transformers torch
```

---

## Run

```bash
python transformer_text_classification_demo.py
```

What the script does:

1. Loads a pre-trained Transformer model.
2. Runs batch predictions on sample customer feedback.
3. Starts real-time mode so you can enter your own text.

Type `exit` to stop real-time mode.
