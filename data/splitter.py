"""
Train / test split utilities.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import TASK1_TEST_SIZE, TASK2_TEST_SIZE, TASK3_TEST_SIZE, SEED


def stratified_sample(df: pd.DataFrame, n: int, label_col: str,
                      seed: int = SEED) -> pd.DataFrame:
    """Balanced stratified sample: take min(n/k, count) per class."""
    k = df[label_col].nunique()
    per_class = n // k
    samples = []
    for label, group in df.groupby(label_col):
        take = min(per_class, len(group))
        samples.append(group.sample(n=take, random_state=seed))
    result = pd.concat(samples).sample(frac=1, random_state=seed)
    return result.head(n)


def split_task1(df: pd.DataFrame, test_size: int = TASK1_TEST_SIZE):
    """Sample a balanced test set for Task 1 (classification)."""
    test = stratified_sample(df, test_size, "label")
    return test


def split_task2(df: pd.DataFrame, test_size: int = TASK2_TEST_SIZE,
                seed: int = SEED):
    """Split MentalChat16K into train (for LoRA) and test (for eval)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    return train_df, test_df


def split_task3(df: pd.DataFrame, test_size: int = TASK3_TEST_SIZE,
                seed: int = SEED):
    """Sample test set for Task 3 (medical QA)."""
    if len(df) <= test_size:
        return df
    return df.sample(n=test_size, random_state=seed)
