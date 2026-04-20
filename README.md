# KannadaSentiment – Hybrid Kannada Sentiment Analysis

KannadaSentiment is an end-to-end sentiment analysis pipeline for **Kannada** text, built around a **hybrid ensemble** of:

- **MuRIL BERT** embeddings (Hugging Face `google/muril-base-cased`)
- **Character-level TF‑IDF** features
- **Logistic Regression** classifiers with a **soft-voting ensemble**

The project includes:

- A reproducible **training pipeline** with data cleaning, feature extraction, model training, evaluation, and artifact generation.
- A beautiful **Streamlit dashboard** for interactive single-text and batch sentiment prediction.
- A **PDF report generator** (ReportLab) for academic or project documentation.
- **Git LFS support** for large model artifacts (`*.pkl` files).

Target labels: `happy`, `sad`, `angry`, `fear`, `disgust`, `neutral`.

---

## Project Structure

- `agents/`
  - `data_agent.py` – loads and cleans the Kannada Excel dataset.
  - `feature_agent.py` – builds TF‑IDF and MuRIL embeddings.
  - `training_agent.py` – trains and saves the TF‑IDF and MuRIL classifiers.
  - `evaluation_agent.py` – evaluates models, generates metrics and confusion matrix.
  - `prediction_agent.py` – loads saved models and runs inference.
  - `orchestrator_agent.py` – orchestrates the full training and prediction pipelines.
- `app/dashboard.py` – Streamlit dashboard for real-time and batch predictions.
- `train.py` – command-line entry point for training.
- `generate_report.py` – generates a multi-page PDF report in `artifacts/`.
- `artifacts/` – evaluation outputs (`confusion_matrix.png`, `classification_report.csv`, `sentiment_report.pdf`).
- `models/` – saved model artifacts (`*.pkl`, tracked via Git LFS).

---

## Prerequisites

- Python **3.10+** (3.11/3.12/3.13 also work)
- `git` and **Git LFS** installed on your system
- Recommended: a virtual environment (e.g. `venv` or `conda`)

Install Git LFS globally (if you haven’t already):

```bash
git lfs install
```

---

## Setup & Installation

From the `KannadaSentiment/` directory:

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install Python dependencies**:

   If you prefer `setup.py`:

   ```bash
   python setup.py
   ```

   Or, if you are using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Initialize git and enable LFS in this repo**:

   ```bash
   git init
   git lfs install
   ```

   The included `.gitattributes` file is already configured to track all `*.pkl` model files with Git LFS:

   ```gitattributes
   *.pkl filter=lfs diff=lfs merge=lfs -text
   ```

---

## Running the Training Pipeline

Place your dataset as an Excel file named `dataset.xlsx` in the project root, with the following columns:

- `kannada_text` – the input text in Kannada
- `sentiment` – one of: `happy`, `sad`, `angry`, `fear`, `disgust`, `neutral`

Then run:

```bash
python train.py
```

This will:

- Clean and preprocess the dataset.
- Extract TF‑IDF and MuRIL features.
- Train the TF‑IDF and MuRIL classifiers.
- Evaluate using a weighted ensemble (35% TF‑IDF, 65% MuRIL).
- Save models and supporting artifacts into:
  - `models/` – `model_tfidf.pkl`, `model_muril.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`
  - `artifacts/` – `confusion_matrix.png`, `classification_report.csv`

The script prints:

- Final **accuracy** and **macro F1** on the test set.
- 5‑fold cross‑validation scores for both TF‑IDF and MuRIL models (to detect over/under‑fitting).

---

## Running the Streamlit Dashboard

After training (so that the `models/` directory is populated), start the UI:

```bash
streamlit run app/dashboard.py
```

The dashboard provides:

- **Realtime prediction** – enter a single Kannada sentence and see:
  - Predicted sentiment with color-coded card.
  - Confidence score as a progress bar.
  - Cleaned text preview.
- **Batch prediction** – upload an `.xlsx` file with a `kannada_text` column:
  - View results in a color-coded table.
  - Download predictions as `predictions.xlsx`.
- **Metrics view** – confusion matrix and classification report table derived from the latest training run.

---

## Generating the PDF Report

To build a polished PDF report (useful for academic submissions, project demos, or documentation):

```bash
python generate_report.py
```

This creates `artifacts/sentiment_report.pdf` containing:

- Cover page with project title and summary metrics.
- Project overview and dataset description.
- System architecture diagram (data → agents → ensemble → prediction).
- Quantitative results and confusion matrix.
- Conclusions and suggested future work.

---

## Git LFS & Large Model Files

Model artifacts in `models/` are binary `*.pkl` files and can easily exceed GitHub’s default file size limits.  
This repository uses **Git LFS** to store them efficiently:

- The `.gitattributes` file includes:

  ```gitattributes
  *.pkl filter=lfs diff=lfs merge=lfs -text
  ```

- After initializing your repository and committing the models:

  ```bash
  git add .gitattributes models/*.pkl
  git commit -m "Add trained Kannada sentiment models"
  git remote add origin <your-github-repo-url>
  git push origin main
  ```

Git LFS will automatically handle actual storage of large `.pkl` files on GitHub.

---

## Live Deployment

You can deploy the Streamlit app to **Streamlit Community Cloud**, **Render**, **Heroku**, or any similar platform.

- **Live demo URL (placeholder)**:  
  `https://kannada-sentiment-analysis.streamlit.app/`

When deploying, ensure:

- `requirements.txt` includes all Python dependencies (`pandas`, `torch`, `transformers`, `streamlit`, etc.).
- The `models/` directory is available to the app (either committed via Git LFS or downloaded at startup).
- The working directory points to the project root so `app/dashboard.py` can import from `agents/`.

---

## License & Acknowledgements

- Pretrained language model: **MuRIL** (`google/muril-base-cased`) from Hugging Face.
- Libraries: `pandas`, `scikit-learn`, `torch`, `transformers`, `matplotlib`, `seaborn`, `streamlit`, `reportlab`, and others as listed in `requirements.txt`.

Please adjust license information and attribution here to match your institution or organization’s requirements.

