import os
import re
from typing import Dict, List

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class PredictionAgent:
    """
    Agent responsible for loading trained models and performing
    sentiment prediction on new Kannada text inputs.
    """

    _KANNADA_CLEAN_PATTERN = re.compile(r"[^\u0C80-\u0CFF\u0CE6-\u0CEF .,!?]+")

    def __init__(self, model_dir: str = "models") -> None:
        """
        Initialize the PredictionAgent by loading all required artifacts.

        Args:
            model_dir: Directory containing the saved models and encoders.
        """
        self.model_dir: str = model_dir

        model_tfidf_path = os.path.join(self.model_dir, "model_tfidf.pkl")
        model_muril_path = os.path.join(self.model_dir, "model_muril.pkl")
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")

        missing_files = [
            path
            for path in [model_tfidf_path, model_muril_path, vectorizer_path, label_encoder_path]
            if not os.path.exists(path)
        ]
        if missing_files:
            missing_str = ", ".join(missing_files)
            raise FileNotFoundError(
                f"[PredictionAgent] Required model files not found in '{self.model_dir}': {missing_str}"
            )

        print(f"[PredictionAgent] Loading models from: {self.model_dir}")
        self.model_tfidf = joblib.load(model_tfidf_path)
        self.model_muril = joblib.load(model_muril_path)
        self.tfidf_vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

        self.device: torch.device = torch.device("cpu")
        print(f"[PredictionAgent] Using device: {self.device}")

        print("[PredictionAgent] Loading MuRIL tokenizer and model")
        self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.model = AutoModel.from_pretrained("google/muril-base-cased", torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

        print("[PredictionAgent] Models loaded successfully")

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean a single text string by keeping only Kannada characters and spaces.

        The allowed Unicode range is U+0C80–U+0CFF.

        Args:
            text: Input text to be cleaned.

        Returns:
            Cleaned text containing only Kannada characters and spaces.
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        cleaned = PredictionAgent._KANNADA_CLEAN_PATTERN.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 3:
            return ""
        return cleaned

    def _extract_muril_embeddings(
        self,
        texts: List[str],
        batch_size: int = 2,
        max_length: int = 64,
    ) -> np.ndarray:
        """
        Extract MuRIL [CLS] embeddings for a list of texts.

        Args:
            texts: List of input texts.
            batch_size: Number of texts per batch.
            max_length: Maximum token length.

        Returns:
            A NumPy array of shape (num_texts, hidden_size) containing embeddings.
        """
        if not texts:
            hidden_size = int(self.model.config.hidden_size)
            return np.zeros((0, hidden_size), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    f"[PredictionAgent] MuRIL embeddings batch {batch_idx + 1}/{num_batches} "
                    f"(texts {start}-{end - 1})"
                )

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings.astype(np.float32))

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    def predict_texts(self, texts: List[str]) -> List[Dict[str, object]]:
        """
        Predict sentiments for a list of input texts using a soft-voting ensemble.

        Args:
            texts: List of raw input texts.

        Returns:
            List of dictionaries, each containing:
                - 'text': Original text.
                - 'cleaned': Cleaned text.
                - 'sentiment': Predicted sentiment label.
                - 'confidence': Confidence percentage (0–100).
        """
        if not texts:
            return []

        print(f"[PredictionAgent] Predicting sentiments for {len(texts)} texts")

        cleaned_texts: List[str] = [self.clean(t) for t in texts]

        print("[PredictionAgent] Transforming texts with TF-IDF vectorizer")
        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts)

        print("[PredictionAgent] Extracting MuRIL embeddings for prediction")
        muril_embeddings = self._extract_muril_embeddings(cleaned_texts, batch_size=2, max_length=64)

        print("[PredictionAgent] Computing probabilities from both models")
        proba_tfidf = self.model_tfidf.predict_proba(tfidf_features)
        proba_muril = self.model_muril.predict_proba(muril_embeddings)

        avg_proba = (0.35 * proba_tfidf) + (0.65 * proba_muril)
        pred_indices = np.argmax(avg_proba, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_indices)

        results: List[Dict[str, object]] = []
        for original, cleaned, label, probs in zip(texts, cleaned_texts, pred_labels, avg_proba):
            confidence = float(np.max(probs) * 100.0)
            results.append(
                {
                    "text": original,
                    "cleaned": cleaned,
                    "sentiment": str(label),
                    "confidence": confidence,
                }
            )

        print("[PredictionAgent] Prediction complete")
        return results

    def save_to_excel(self, results: List[Dict[str, object]], path: str = "artifacts/predictions.xlsx") -> None:
        """
        Save prediction results to an Excel file.

        Args:
            results: List of prediction result dictionaries.
            path: Output Excel file path.
        """
        import pandas as pd

        if not results:
            print("[PredictionAgent] No results to save to Excel.")
            return

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df = pd.DataFrame(results)
        df.to_excel(path, index=False)
        print(f"[PredictionAgent] Predictions saved to: {path}")

