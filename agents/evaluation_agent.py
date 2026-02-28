import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class EvaluationAgent:
    """
    Agent responsible for evaluating trained models and producing metrics
    and visual artifacts such as the confusion matrix.
    """

    def __init__(self, output_dir: str = "artifacts") -> None:
        """
        Initialize the EvaluationAgent.

        Args:
            output_dir: Directory where evaluation artifacts will be stored.
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[EvaluationAgent] Artifacts will be saved to: {self.output_dir}")

    def evaluate(
        self,
        model_tfidf,
        model_muril,
        tfidf_test,
        muril_test: np.ndarray,
        y_true: np.ndarray,
        label_encoder,
    ) -> Dict[str, float]:
        """
        Evaluate models using a soft-voting ensemble of TF-IDF and MuRIL models.

        Args:
            model_tfidf: Trained TF-IDF-based classifier.
            model_muril: Trained MuRIL-based classifier.
            tfidf_test: TF-IDF features for the test set.
            muril_test: MuRIL embeddings for the test set.
            y_true: True encoded labels for the test set.
            label_encoder: Fitted LabelEncoder instance.

        Returns:
            Dictionary containing 'accuracy' and 'macro_f1'.
        """
        print("[EvaluationAgent] Starting evaluation")

        if tfidf_test.shape[0] != muril_test.shape[0] or tfidf_test.shape[0] != y_true.shape[0]:
            raise ValueError(
                "[EvaluationAgent] Mismatched number of samples between TF-IDF, MuRIL, and labels."
            )

        print("[EvaluationAgent] Computing predict_proba for TF-IDF model")
        proba_tfidf = model_tfidf.predict_proba(tfidf_test)

        print("[EvaluationAgent] Computing predict_proba for MuRIL model")
        proba_muril = model_muril.predict_proba(muril_test)

        print("[EvaluationAgent] Weighted ensemble (35% TF-IDF, 65% MuRIL)")
        avg_proba = (0.35 * proba_tfidf) + (0.65 * proba_muril)
        y_pred = np.argmax(avg_proba, axis=1)

        print("[EvaluationAgent] Calculating metrics")
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        class_names: List[str] = list(label_encoder.classes_)

        print(f"[EvaluationAgent] Accuracy: {accuracy:.4f}")
        print(f"[EvaluationAgent] Weighted F1: {f1_weighted:.4f}")
        print(f"[EvaluationAgent] Macro F1: {f1_macro:.4f}")

        print("[EvaluationAgent] Classification report:")
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        print(report_str)

        # Save classification report as a CSV for downstream use (dashboard, report).
        print("[EvaluationAgent] Saving classification report to CSV")
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})
        report_csv_path = os.path.join(self.output_dir, "classification_report.csv")
        report_df.to_csv(report_csv_path, index=False)
        print(f"[EvaluationAgent] Classification report saved to: {report_csv_path}")

        print("[EvaluationAgent] Plotting confusion matrix")
        self._plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=class_names)

        print("[EvaluationAgent] Evaluation completed")
        return {"accuracy": float(accuracy), "macro_f1": float(f1_macro)}

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
        """
        Plot and save the confusion matrix as a heatmap.

        Args:
            y_true: True encoded labels.
            y_pred: Predicted encoded labels.
            class_names: List of class names corresponding to encoded labels.
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Hybrid Kannada Sentiment Analysis - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[EvaluationAgent] Confusion matrix saved to: {cm_path}")

