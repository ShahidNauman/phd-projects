from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_dataframe(
    df: pd.DataFrame,
    *,
    group_column: str = "group_id",
    seed: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not {"text", "label", group_column}.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain text, label, and {group_column} columns"
        )

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if df[group_column].nunique() < 3:
        train, rest = train_test_split(
            df, train_size=train_size, stratify=df["label"], random_state=seed
        )
        validation_fraction = validation_size / (1.0 - train_size)
        validation, test = train_test_split(
            rest,
            train_size=validation_fraction,
            stratify=_stratify_or_none(rest["label"]),
            random_state=seed,
        )
        return _clean(train), _clean(validation), _clean(test)

    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, rest_idx = next(splitter.split(df, df["label"], groups=df[group_column]))
    train = df.iloc[train_idx]
    rest = df.iloc[rest_idx]

    validation_fraction = validation_size / (1.0 - train_size)
    rest_splitter = GroupShuffleSplit(
        n_splits=1, train_size=validation_fraction, random_state=seed + 1
    )
    validation_idx, test_idx = next(
        rest_splitter.split(rest, rest["label"], groups=rest[group_column])
    )
    validation = rest.iloc[validation_idx]
    test = rest.iloc[test_idx]
    return _clean(train), _clean(validation), _clean(test)


def _stratify_or_none(labels: pd.Series):
    counts = labels.value_counts()
    return labels if len(counts) > 1 and np.all(counts >= 2) else None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)
