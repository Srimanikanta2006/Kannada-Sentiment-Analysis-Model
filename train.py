from typing import Dict

from agents.orchestrator_agent import OrchestratorAgent


BANNER = r"""
============================================================
   KANNADA SENTIMENT ANALYSIS - TRAINING STARTED
============================================================
"""


def main() -> None:
    """
    Entry point for running the full training pipeline.

    Steps:
        1. Initialize OrchestratorAgent with the dataset path.
        2. Run the training pipeline.
        3. Print a final summary with key metrics and next steps.
    """
    print(BANNER)

    DATASET_FILE = "dataset.xlsx"

    try:
        orchestrator = OrchestratorAgent(data_path=DATASET_FILE)
        metrics: Dict[str, float] = orchestrator.run_training_pipeline()

        accuracy = metrics.get("accuracy", 0.0)
        macro_f1 = metrics.get("macro_f1", 0.0)

        summary = f"""
============================================================
                    TRAINING SUMMARY
------------------------------------------------------------
 Accuracy achieved      : {accuracy:.4f}
 Macro F1 score         : {macro_f1:.4f}
 Models saved in        : models
 Confusion matrix saved : artifacts/confusion_matrix.png

 Next steps:
   Now run: streamlit run app/dashboard.py
============================================================
"""
        print(summary)

    except FileNotFoundError as exc:
        print(f"[ERROR] Dataset file not found: {exc}")
    except Exception as exc:
        print(f"[ERROR] Training pipeline failed: {exc}")


if __name__ == "__main__":
    main()

