from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .splits import split_dataframe


LABEL_MAP = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "non-sarcastic": 0,
    "nonsarcastic": 0,
    "not sarcastic": 0,
    "regular": 0,
    "sarcastic": 1,
    "ironic": 1,
}


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing label")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    key = str(value).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    raise ValueError(f"Unsupported sarcasm label: {value!r}")


def normalize_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        raise ValueError("Empty text row")
    return text


def write_splits(df: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = split_dataframe(df, group_column="group_id", seed=seed)
    for name, split_df in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        split_df.to_json(
            output_dir / f"{name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
    metadata = {
        "rows": int(len(df)),
        "train": int(len(train)),
        "validation": int(len(validation)),
        "test": int(len(test)),
        "label_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def prepare_csv_dataset(
    csv_path: Path,
    output_dir: Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
    group_column: str | None = None,
    seed: int = 42,
) -> None:
    df = pd.read_csv(csv_path)
    records = pd.DataFrame(
        {
            "text": df[text_column].map(normalize_text),
            "label": df[label_column].map(normalize_label),
            "source": "csv",
        }
    )
    if group_column and group_column in df.columns:
        records["group_id"] = df[group_column].astype(str)
    else:
        records["group_id"] = [f"csv-{idx}" for idx in range(len(records))]
    write_splits(records, output_dir, seed)


def prepare_hf_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    config_name: str | None = None,
    split: str = "train",
    text_column: str = "headline",
    label_column: str = "is_sarcastic",
    seed: int = 42,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets to use prepare-hf: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, config_name, split=split)
    df = dataset.to_pandas()
    records = pd.DataFrame(
        {
            "text": df[text_column].map(normalize_text),
            "label": df[label_column].map(normalize_label),
            "source": dataset_name,
            "group_id": [f"{dataset_name}-{idx}" for idx in range(len(df))],
        }
    )
    write_splits(records, output_dir, seed)


def prepare_amazon_corpus(
    corpus_path: Path, output_dir: Path, *, seed: int = 42
) -> None:
    rows = list(_iter_amazon_rows(corpus_path))
    if not rows:
        raise ValueError(
            "No Amazon corpus rows were found. Extract Ironic.rar and Regular.rar first, "
            "or convert the corpus to CSV and use prepare-csv."
        )
    write_splits(pd.DataFrame(rows), output_dir, seed)


def _iter_amazon_rows(corpus_path: Path):
    base = _resolve_corpus_root(corpus_path)
    candidates = (
        list(base.rglob("*.txt"))
        + list(base.rglob("*.csv"))
        + list(base.rglob("*.jsonl"))
    )
    for path in candidates:
        if path.name.lower() in {"readme", "sarcasm_lines.txt", "file_pairing.txt"}:
            continue
        label = _label_from_path(path)
        if label is None:
            continue
        for idx, text in enumerate(_read_text_records(path)):
            yield {
                "text": normalize_text(text),
                "label": label,
                "source": "amazon",
                "group_id": _group_from_path(path, idx),
            }


def _resolve_corpus_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix.lower() == ".zip":
        extract_dir = path.with_suffix("")
        if not extract_dir.exists():
            with zipfile.ZipFile(path) as archive:
                archive.extractall(extract_dir)
        children = list(extract_dir.iterdir())
        return (
            children[0] if len(children) == 1 and children[0].is_dir() else extract_dir
        )
    raise ValueError(f"Unsupported corpus path: {path}")


def _label_from_path(path: Path) -> int | None:
    lowered = " ".join(part.lower() for part in path.parts)
    if "ironic" in lowered or "sarcastic" in lowered:
        return 1
    if "regular" in lowered or "non" in lowered:
        return 0
    return None


def _group_from_path(path: Path, idx: int) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"^(ironic|regular|sarcastic|non[-_ ]?sarcastic)[-_ ]*", "", stem)
    return stem or f"{path.stem}-{idx}"


def _read_text_records(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        column = "text" if "text" in df.columns else df.columns[0]
        yield from df[column].dropna().astype(str)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                row = json.loads(line)
                yield row.get("text") or row.get("review") or row.get("headline")
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
    yield from chunks if chunks else [content]
