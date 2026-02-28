import os
import time
from typing import Dict, List

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from agents.data_agent import DataAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feature_agent import FeatureAgent
from agents.prediction_agent import PredictionAgent
from agents.training_agent import TrainingAgent


class OrchestratorAgent:
    """
    High-level agent that orchestrates the full training and prediction pipelines.
    """

    def __init__(self, data_path: str, model_dir: str = "models", output_dir: str = "artifacts") -> None:
        """
        Initialize the OrchestratorAgent.

        Args:
            data_path: Path to the dataset Excel file.
            model_dir: Directory where models will be saved/loaded.
            output_dir: Directory for evaluation artifacts.
        """
        self.data_path: str = data_path
        self.model_dir: str = model_dir
        self.output_dir: str = output_dir

        self.data_agent = DataAgent(file_path=data_path)
        self.feature_agent = FeatureAgent()
        self.training_agent = TrainingAgent(model_save_dir=model_dir)
        self.evaluation_agent = EvaluationAgent(output_dir=output_dir)

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def run_training_pipeline(self) -> Dict[str, float]:
        """
        Run the full training pipeline:

        Step 1: DataAgent - load_and_clean() + get_texts_and_labels()
        Step 2: train_test_split (with stratify when possible).
        Step 3: FeatureAgent - TF-IDF + MuRIL feature extraction.
        Step 4: TrainingAgent - encode_labels, train, save.
        Step 5: EvaluationAgent - evaluate on test set.

        Returns:
            Dictionary with keys: 'accuracy', 'macro_f1'.

        Raises:
            Exception: Propagates any critical failure from the steps.
        """
        overall_start = time.time()
        print("[OrchestratorAgent] Starting training pipeline")

        # Step 1: Data loading and cleaning
        step_start = time.time()
        print("[OrchestratorAgent][Step 1] Loading and cleaning data")
        try:
            _ = self.data_agent.load_and_clean()
            texts, labels = self.data_agent.get_texts_and_labels()
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR][Step 1] Failed to load and clean data: {exc}")
            raise
        print(
            f"[OrchestratorAgent][Step 1] Completed in {time.time() - step_start:.2f}s "
            f"with {len(texts)} samples"
        )

        if len(texts) < 2:
            raise ValueError("[OrchestratorAgent] Not enough samples for training (need at least 2).")

        # Step 2: Train/test split
        step_start = time.time()
        print("[OrchestratorAgent][Step 2] Splitting data into train and test sets")
        texts_arr = np.array(texts)
        labels_arr = np.array(labels)

        stratify_labels: List[str] | None = labels_arr.tolist()
        unique, counts = np.unique(labels_arr, return_counts=True)
        if np.any(counts < 2) or unique.size < 2:
            print(
                "[OrchestratorAgent][Step 2] Some classes have <2 samples or only one class present; "
                "disabling stratified split."
            )
            stratify_labels = None

        try:
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts_arr,
                labels_arr,
                test_size=0.2,
                random_state=42,
                stratify=stratify_labels,
            )
        except ValueError as exc:
            print(
                f"[OrchestratorAgent][WARNING][Step 2] Stratified split failed ({exc}); "
                "falling back to non-stratified split."
            )
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts_arr,
                labels_arr,
                test_size=0.2,
                random_state=42,
                stratify=None,
            )
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR][Step 2] Failed to split data: {exc}")
            raise

        print(
            f"[OrchestratorAgent][Step 2] Completed in {time.time() - step_start:.2f}s "
            f"(train={len(X_train_texts)}, test={len(X_test_texts)})"
        )

        # Step 3: Feature extraction
        step_start = time.time()
        print("[OrchestratorAgent][Step 3] Extracting TF-IDF and MuRIL features")
        try:
            tfidf_train = self.feature_agent.fit_transform_tfidf(X_train_texts.tolist())
            tfidf_test = self.feature_agent.transform_tfidf(X_test_texts.tolist())

            muril_train = self.feature_agent.extract_muril_embeddings(X_train_texts.tolist())
            muril_test = self.feature_agent.extract_muril_embeddings(X_test_texts.tolist())
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR][Step 3] Feature extraction failed: {exc}")
            raise

        print(
            f"[OrchestratorAgent][Step 3] Completed in {time.time() - step_start:.2f}s "
            f"(TF-IDF train shape={tfidf_train.shape}, MuRIL train shape={muril_train.shape})"
        )

        # Step 4: Training
        step_start = time.time()
        print("[OrchestratorAgent][Step 4] Training models")
        try:
            y_train_encoded = self.training_agent.encode_labels(y_train.tolist())
            y_test_encoded = self.training_agent.label_encoder.transform(y_test.tolist())

            self.training_agent.train(
                tfidf_features=tfidf_train,
                muril_embeddings=muril_train,
                encoded_labels=y_train_encoded,
            )

            self.training_agent.save(self.feature_agent)

            print("[OrchestratorAgent] Running 5-fold cross-validation on train set")
            tfidf_cv = cross_val_score(
                self.training_agent.model_tfidf,
                tfidf_train,
                y_train_encoded,
                cv=5,
                scoring="accuracy",
            )
            muril_cv = cross_val_score(
                self.training_agent.model_muril,
                muril_train,
                y_train_encoded,
                cv=5,
                scoring="accuracy",
            )
            print(
                f"TF-IDF 5-fold CV: {tfidf_cv.mean():.4f} (+/- {tfidf_cv.std():.4f})"
            )
            print(f"MuRIL  5-fold CV: {muril_cv.mean():.4f} (+/- {muril_cv.std():.4f})")
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR][Step 4] Training or saving models failed: {exc}")
            raise

        print(
            f"[OrchestratorAgent][Step 4] Completed in {time.time() - step_start:.2f}s "
            "and models saved."
        )

        # Step 5: Evaluation
        step_start = time.time()
        print("[OrchestratorAgent][Step 5] Evaluating models on test set")
        try:
            metrics = self.evaluation_agent.evaluate(
                model_tfidf=self.training_agent.model_tfidf,
                model_muril=self.training_agent.model_muril,
                tfidf_test=tfidf_test,
                muril_test=muril_test,
                y_true=y_test_encoded,
                label_encoder=self.training_agent.label_encoder,
            )
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR][Step 5] Evaluation failed: {exc}")
            raise

        print(
            f"[OrchestratorAgent][Step 5] Completed in {time.time() - step_start:.2f}s "
            f"with metrics: {metrics}"
        )
        print(f"[OrchestratorAgent] Training pipeline completed in {time.time() - overall_start:.2f}s")

        return metrics

    def run_prediction_pipeline(self, texts: List[str]) -> List[Dict[str, object]]:
        """
        Run the prediction pipeline using the PredictionAgent.

        Args:
            texts: List of input texts to predict.

        Returns:
            List of prediction result dictionaries as produced by PredictionAgent.
        """
        print("[OrchestratorAgent] Starting prediction pipeline")
        try:
            predictor = PredictionAgent(model_dir=self.model_dir)
            results = predictor.predict_texts(texts)
            print("[OrchestratorAgent] Prediction pipeline completed successfully")
            return results
        except Exception as exc:
            print(f"[OrchestratorAgent][ERROR] Prediction pipeline failed: {exc}")
            raise

