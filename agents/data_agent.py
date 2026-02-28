import os
import re
from typing import List, Tuple, Optional

import pandas as pd


class DataAgent:
    """
    Agent responsible for loading and cleaning the Kannada sentiment dataset.

    The dataset is expected to contain at least two columns:
    - kannada_text (by default)
    - sentiment (by default)
    """

    _KANNADA_CLEAN_PATTERN = re.compile(r"[^\u0C80-\u0CFF\u0CE6-\u0CEF .,!?]+")

    def __init__(
        self,
        file_path: str,
        text_col: str = "kannada_text",
        label_col: str = "sentiment",
    ) -> None:
        """
        Initialize the DataAgent.

        Args:
            file_path: Path to the Excel dataset file.
            text_col: Name of the text column in the dataset.
            label_col: Name of the label/target column in the dataset.
        """
        self.file_path: str = file_path
        self.text_col: str = text_col
        self.label_col: str = label_col
        self._clean_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _clean_kannada_text(text: str) -> str:
        """
        Clean a single text string by keeping only Kannada characters and spaces.

        The allowed ranges are U+0C80–U+0CFF (Kannada), U+0CE6–U+0CEF (numerals),
        spaces, and basic punctuation (.,!?). Returns empty string if cleaned
        text has fewer than 3 characters.

        Args:
            text: Input text to be cleaned.

        Returns:
            Cleaned text containing only Kannada characters and spaces.
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        # Remove all characters outside Kannada block, numerals, spaces, basic punctuation.
        cleaned = DataAgent._KANNADA_CLEAN_PATTERN.sub(" ", text)
        # Normalize whitespace.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 3:
            return ""
        return cleaned

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load the dataset from Excel and apply a series of cleaning steps.

        Steps:
            1. Read the Excel file using engine='openpyxl'.
            2. Drop rows where the text or label is NaN.
            3. Normalize labels (strip whitespace and lowercase).
            4. Clean text to keep only Kannada characters and spaces.
            5. Remove rows that are empty after cleaning.
            6. Remove any label class that appears fewer than 2 times.
            7. Print class distribution after cleaning.

        Returns:
            A cleaned pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If required columns are missing or dataset becomes empty.
        """
        print("[DataAgent] Starting load_and_clean()")

        if not os.path.exists(self.file_path):
            message = f"Dataset file not found at path: {self.file_path}"
            print(f"[DataAgent][ERROR] {message}")
            raise FileNotFoundError(message)

        print(f"[DataAgent] Reading Excel file: {self.file_path}")
        try:
            df = pd.read_excel(self.file_path, engine="openpyxl")
        except Exception as exc:
            print(f"[DataAgent][ERROR] Failed to read Excel file: {exc}")
            raise

        if df.empty:
            message = "Loaded dataset is empty."
            print(f"[DataAgent][ERROR] {message}")
            raise ValueError(message)

        print(f"[DataAgent] Initial dataset shape: {df.shape}")

        # Validate required columns.
        missing_cols = [col for col in [self.text_col, self.label_col] if col not in df.columns]
        if missing_cols:
            message = f"Required columns missing from dataset: {missing_cols}"
            print(f"[DataAgent][ERROR] {message}")
            raise ValueError(message)

        # Drop rows with NaN in text or label.
        print("[DataAgent] Dropping rows with NaN in text or label columns")
        df = df.dropna(subset=[self.text_col, self.label_col])
        print(f"[DataAgent] Shape after dropping NaNs: {df.shape}")

        # Normalize labels.
        print("[DataAgent] Normalizing labels (strip + lowercase)")
        df[self.label_col] = df[self.label_col].astype(str).str.strip().str.lower()
        print(f"[DataAgent] Unique labels after normalization: {sorted(df[self.label_col].unique())}")

        # Clean text.
        print(
            "[DataAgent] Cleaning Kannada text (U+0C80–U+0CFF, numerals U+0CE6–U+0CEF, .,!?)"
        )
        df[self.text_col] = df[self.text_col].astype(str).apply(self._clean_kannada_text)

        # Remove rows that became empty after cleaning.
        print("[DataAgent] Removing rows that are empty after cleaning")
        df = df[df[self.text_col].str.len() > 0]
        print(f"[DataAgent] Shape after removing empty texts: {df.shape}")

        if df.empty:
            message = "All rows were removed after text cleaning; dataset is empty."
            print(f"[DataAgent][ERROR] {message}")
            raise ValueError(message)

        # Remove any class label that appears fewer than 2 times.
        print("[DataAgent] Removing labels that appear fewer than 2 times")
        label_counts = df[self.label_col].value_counts()
        valid_labels = label_counts[label_counts >= 2].index
        invalid_labels = label_counts[label_counts < 2].index.tolist()

        if invalid_labels:
            print(f"[DataAgent] Labels removed due to insufficient samples (<2): {invalid_labels}")

        df = df[df[self.label_col].isin(valid_labels)]
        print(f"[DataAgent] Shape after removing rare labels: {df.shape}")

        if df.empty:
            message = "All rows were removed after filtering rare labels; dataset is empty."
            print(f"[DataAgent][ERROR] {message}")
            raise ValueError(message)

        # Final class distribution.
        print("[DataAgent] Class distribution after cleaning:")
        print(df[self.label_col].value_counts().to_string())

        self._clean_df = df.reset_index(drop=True)
        print("[DataAgent] load_and_clean() completed successfully")
        return self._clean_df

    def get_texts_and_labels(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve lists of texts and labels from the cleaned dataset.

        If the dataset has not yet been loaded and cleaned, this will call
        load_and_clean() automatically.

        Returns:
            A tuple (list_of_texts, list_of_labels).

        Raises:
            ValueError: If the dataset remains empty after cleaning.
        """
        if self._clean_df is None:
            print("[DataAgent] No cached cleaned DataFrame found; calling load_and_clean()")
            self.load_and_clean()

        assert self._clean_df is not None  # for type checkers

        texts: List[str] = self._clean_df[self.text_col].astype(str).tolist()
        labels: List[str] = self._clean_df[self.label_col].astype(str).tolist()

        if not texts or not labels:
            message = "Cleaned dataset has no texts or labels."
            print(f"[DataAgent][ERROR] {message}")
            raise ValueError(message)

        print(f"[DataAgent] Returning {len(texts)} texts and {len(labels)} labels")
        return texts, labels

