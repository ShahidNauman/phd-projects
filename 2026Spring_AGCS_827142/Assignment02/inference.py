"""
Inference demo for the Transformer sentiment classifier.

Loads a trained checkpoint (``best_model.pt``) and runs interactive
sentiment analysis on user-supplied sentences.  If no checkpoint exists,
the script trains from scratch with default settings before running inference.

Usage:
    python inference.py
    python inference.py --text "This movie was absolutely fantastic!"
    python inference.py --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from dataset import SimpleTokenizer, SAMPLE_DATA, get_dataloaders
from transformer import TransformerClassifier

LABELS = {0: "NEGATIVE 😞", 1: "POSITIVE 😊"}


def load_model(
    checkpoint_path: str,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    max_len: int = 64,
) -> TransformerClassifier:
    """Instantiate and load weights into a TransformerClassifier.

    Args:
        checkpoint_path: Path to a ``state_dict`` saved with torch.save.
        tokenizer: Fitted SimpleTokenizer (needed for vocab size).
        device: Device to load the model onto.

    Returns:
        TransformerClassifier in evaluation mode.
    """
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_len=max_len,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict(
    text: str,
    model: TransformerClassifier,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    max_len: int = 64,
) -> tuple[str, float]:
    """Predict the sentiment label for a single text input.

    Args:
        text: Raw input sentence.
        model: Trained TransformerClassifier.
        tokenizer: Fitted tokenizer.
        device: Inference device.
        max_len: Maximum sequence length.

    Returns:
        Tuple of (label_string, confidence) where confidence is the
        softmax probability for the predicted class.
    """
    ids = tokenizer.encode(text, max_len)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    padding_mask = (input_ids == 0)

    with torch.no_grad():
        logits = model(input_ids, padding_mask)
        probs = torch.softmax(logits, dim=-1)

    pred_class = probs.argmax(dim=-1).item()
    confidence = probs[0, pred_class].item()
    return LABELS[pred_class], confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer sentiment inference demo")
    parser.add_argument("--text", type=str, default=None, help="Single sentence to classify")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to model checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build tokenizer from the full dataset.
    _, _, tokenizer = get_dataloaders()

    # Train if checkpoint is missing.
    if not os.path.exists(args.checkpoint):
        print("No checkpoint found — training model first …\n")
        import subprocess
        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            print("Training failed. Exiting.")
            sys.exit(1)

    model = load_model(args.checkpoint, tokenizer, device)

    # ---- Single-sentence mode -----------------------------------------------
    if args.text:
        label, conf = predict(args.text, model, tokenizer, device)
        print(f"\nInput   : {args.text}")
        print(f"Sentiment: {label}  (confidence: {conf:.1%})\n")
        return

    # ---- Interactive demo mode ----------------------------------------------
    print("\n" + "=" * 60)
    print("  Transformer Sentiment Analysis — Interactive Demo")
    print("=" * 60)
    print("Type a sentence and press Enter to get its sentiment.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Show a few built-in examples first.
    demo_sentences = [
        "The film was an absolute masterpiece with stunning visuals.",
        "I wasted two hours watching this terrible movie.",
        "A decent story but the pacing was a bit slow at times.",
    ]
    print("── Built-in examples ──")
    for sent in demo_sentences:
        label, conf = predict(sent, model, tokenizer, device)
        print(f"  [{label} {conf:.0%}]  {sent}")

    print("\n── Try your own ──")
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break
        label, conf = predict(user_input, model, tokenizer, device)
        print(f"  Sentiment: {label}  (confidence: {conf:.1%})")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
