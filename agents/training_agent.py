import os
from typing import List

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class TrainingAgent:
    """
    Agent responsible for training and saving classification models
    for both TF-IDF features and MuRIL embeddings.
    """

    def __init__(self, model_save_dir: str = "models") -> None:
        """
        Initialize the TrainingAgent.

        Args:
            model_save_dir: Directory where trained models and artifacts will be saved.
        """
        self.model_save_dir: str = model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
        print(f"[TrainingAgent] Models will be saved to: {self.model_save_dir}")

        self.model_tfidf = LogisticRegression(
            C=10.0,
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.model_muril = LogisticRegression(
            C=10.0,
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Fit the label encoder on the provided labels and return encoded integers.

        Args:
            labels: List of string labels.

        Returns:
            NumPy array of encoded label integers.
        """
        if not labels:
            raise ValueError("[TrainingAgent] No labels provided to encode_labels().")

        print(f"[TrainingAgent] Encoding {len(labels)} labels")
        encoded = self.label_encoder.fit_transform(labels)
        print(
            "[TrainingAgent] Label mapping: "
            f"{dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}"
        )
        return encoded

    def train(
        self,
        tfidf_features: csr_matrix,
        muril_embeddings: np.ndarray,
        encoded_labels: np.ndarray,
    ) -> None:
        """
        Train logistic regression models on TF-IDF features and MuRIL embeddings.

        Args:
            tfidf_features: Sparse matrix of TF-IDF features.
            muril_embeddings: Dense matrix of MuRIL embeddings.
            encoded_labels: Encoded label integers.
        """
        if tfidf_features.shape[0] != muril_embeddings.shape[0] or tfidf_features.shape[0] != encoded_labels.shape[0]:
            raise ValueError(
                "[TrainingAgent] Mismatched number of samples between TF-IDF, MuRIL, and labels."
            )

        print("[TrainingAgent] Training logistic regression on TF-IDF features")
        self.model_tfidf.fit(tfidf_features, encoded_labels)
        print("[TrainingAgent] TF-IDF model training complete")

        print("[TrainingAgent] Training logistic regression on MuRIL embeddings")
        self.model_muril.fit(muril_embeddings, encoded_labels)
        print("[TrainingAgent] MuRIL model training complete")

    def save(self, feature_agent) -> None:
        """
        Save trained models, TF-IDF vectorizer, and label encoder to disk.

        Args:
            feature_agent: Instance of FeatureAgent containing the fitted TF-IDF vectorizer.
        """
        os.makedirs(self.model_save_dir, exist_ok=True)

        tfidf_model_path = os.path.join(self.model_save_dir, "model_tfidf.pkl")
        muril_model_path = os.path.join(self.model_save_dir, "model_muril.pkl")
        vectorizer_path = os.path.join(self.model_save_dir, "tfidf_vectorizer.pkl")
        label_encoder_path = os.path.join(self.model_save_dir, "label_encoder.pkl")

        joblib.dump(self.model_tfidf, tfidf_model_path)
        print(f"[TrainingAgent] Saved TF-IDF model to: {tfidf_model_path}")

        joblib.dump(self.model_muril, muril_model_path)
        print(f"[TrainingAgent] Saved MuRIL model to: {muril_model_path}")

        joblib.dump(feature_agent.vectorizer, vectorizer_path)
        print(f"[TrainingAgent] Saved TF-IDF vectorizer to: {vectorizer_path}")

        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"[TrainingAgent] Saved LabelEncoder to: {label_encoder_path}")

