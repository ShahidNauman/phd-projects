from __future__ import annotations

from pathlib import Path

from .modeling import load_tokenizer_and_model
from .templates import format_judge_text


def predict_texts(
    model_path: Path,
    texts: list[str],
    threshold: float = 0.5,
    max_length: int = 192,
    prompt_template: str | None = None,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch before running prediction.") from exc

    tokenizer, model = load_tokenizer_and_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    judge_texts = [format_judge_text(text, prompt_template) for text in texts]
    encoded = tokenizer(
        judge_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    return [
        {
            "text": text,
            "sarcasm_probability": float(probability),
            "label": "sarcastic" if probability >= threshold else "non_sarcastic",
        }
        for text, probability in zip(texts, probabilities)
    ]
