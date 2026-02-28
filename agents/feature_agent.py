from typing import List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


class FeatureAgent:
    """
    Agent responsible for feature extraction using:
    - TF-IDF character-level features.
    - MuRIL transformer embeddings.
    """

    def __init__(self, model_name: str = "google/muril-base-cased", max_features: int = 10000) -> None:
        """
        Initialize the FeatureAgent.

        Args:
            model_name: Name of the MuRIL model to load.
            max_features: Maximum number of TF-IDF features.
        """
        self.model_name: str = model_name
        self.max_features: int = max_features

        self.device: torch.device = torch.device("cpu")
        print(f"[FeatureAgent] Using device: {self.device}")

        print(f"[FeatureAgent] Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

        print(f"[FeatureAgent] Initializing TfidfVectorizer with max_features={max_features}")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=2,
        )

    def fit_transform_tfidf(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer on training texts and return the feature matrix.

        Args:
            texts: List of training texts.

        Returns:
            Sparse TF-IDF feature matrix for the input texts.
        """
        if not texts:
            raise ValueError("[FeatureAgent] No texts provided to fit_transform_tfidf().")

        print(f"[FeatureAgent] Fitting TF-IDF on {len(texts)} texts")
        features = self.vectorizer.fit_transform(texts)
        print("[FeatureAgent] TF-IDF fitting complete")
        return features

    def transform_tfidf(self, texts: List[str]):
        """
        Transform texts using an already-fitted TF-IDF vectorizer.

        Args:
            texts: List of texts to transform.

        Returns:
            Sparse TF-IDF feature matrix for the input texts.
        """
        if not texts:
            raise ValueError("[FeatureAgent] No texts provided to transform_tfidf().")

        print(f"[FeatureAgent] Transforming {len(texts)} texts using TF-IDF")
        features = self.vectorizer.transform(texts)
        print("[FeatureAgent] TF-IDF transformation complete")
        return features

    def extract_muril_embeddings(self, texts: List[str], batch_size: int = 2) -> np.ndarray:
        """
        Extract MuRIL [CLS] embeddings for a list of texts.

        The method processes texts in small batches to remain friendly to
        low-RAM laptops.

        Args:
            texts: List of input texts.
            batch_size: Number of texts per batch.

        Returns:
            A NumPy array of shape (num_texts, hidden_size) containing embeddings.
        """
        if not texts:
            print("[FeatureAgent] No texts provided to extract_muril_embeddings(); returning empty array.")
            hidden_size = int(self.model.config.hidden_size)
            return np.zeros((0, hidden_size), dtype=np.float32)

        print(f"[FeatureAgent] Extracting MuRIL embeddings for {len(texts)} texts with batch_size={batch_size}")
        embeddings_list: List[np.ndarray] = []

        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"[FeatureAgent] Processing batch {batch_idx + 1}/{num_batches} (texts {start}-{end - 1})")

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_embeddings.astype(np.float32))

            # Free RAM as much as possible.
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embeddings = np.vstack(embeddings_list)
        print(f"[FeatureAgent] Finished extracting embeddings; shape={embeddings.shape}")
        return embeddings

