"""
Data loading and preprocessing for three datasets.
"""
import csv
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    SENTIMENT_CSV, MEDQUAD_CSV,
    MENTALCHAT_INTERVIEW, MENTALCHAT_SYNTHETIC,
    SENTIMENT_LABELS, MENTAL_HEALTH_KEYWORDS, SEED,
)


def load_sentiment(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Sentiment Analysis for Mental Health dataset."""
    path = path or SENTIMENT_CSV
    df = pd.read_csv(path)
    # columns: unnamed index, statement, status
    df = df.rename(columns={"statement": "text", "status": "label"})
    df = df[["text", "label"]].dropna()
    return df


def load_mentalchat(path_interview: Optional[Path] = None,
                    path_synthetic: Optional[Path] = None) -> pd.DataFrame:
    """Load MentalChat16K (interview + synthetic combined)."""
    p_int = path_interview or MENTALCHAT_INTERVIEW
    p_syn = path_synthetic or MENTALCHAT_SYNTHETIC
    df_int = pd.read_csv(p_int)
    df_syn = pd.read_csv(p_syn)
    df = pd.concat([df_int, df_syn], ignore_index=True)
    # columns: instruction, input, output
    return df


def load_medquad(path: Optional[Path] = None,
                 mental_health_only: bool = False) -> pd.DataFrame:
    """Load MedQuAD dataset. Optionally filter to mental-health topics."""
    path = path or MEDQUAD_CSV
    df = pd.read_csv(path)
    # columns: question, answer, source, focus_area
    if mental_health_only:
        mask = df["focus_area"].str.lower().apply(
            lambda x: any(kw in x for kw in MENTAL_HEALTH_KEYWORDS)
            if pd.notna(x) else False
        )
        df = df[mask]
    return df
