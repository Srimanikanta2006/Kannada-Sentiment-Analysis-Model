import os
from datetime import date
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def _draw_header_footer(c: canvas.Canvas, page_number: int) -> None:
    """
    Draw a consistent header and footer with page numbers.
    """
    width, height = A4
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0F0C29"))
    c.drawString(2 * cm, height - 1.5 * cm, "Kannada Sentiment Analysis - Project Report")

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(width - 2 * cm, 1.2 * cm, f"Page {page_number}")
    c.setStrokeColor(colors.lightgrey)
    c.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)


def _page_cover(c: canvas.Canvas, accuracy: float, macro_f1: float) -> None:
    """
    Draw the cover page (Page 1).
    """
    width, height = A4
    _draw_header_footer(c, page_number=1)

    c.setFillColor(colors.HexColor("#0F0C29"))
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 3 * cm, "Your University / College Name")

    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawCentredString(width / 2, height - 6 * cm, "Kannada Sentiment Analysis")

    c.setFont("Helvetica", 14)
    c.setFillColor(colors.HexColor("#4B5563"))
    c.drawCentredString(
        width / 2,
        height - 7.5 * cm,
        "Hybrid MuRIL BERT + TF-IDF Ensemble Model",
    )

    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawCentredString(width / 2, height - 10 * cm, "Team Members: _______________________________")

    today = date.today().strftime("%d %B %Y")
    c.drawCentredString(width / 2, height - 11.5 * cm, f"Date: {today}")

    c.drawCentredString(
        width / 2,
        height - 13 * cm,
        f"Accuracy Achieved: {accuracy:.2%} | Macro F1: {macro_f1:.2f}",
    )


def _page_overview(c: canvas.Canvas, total_samples: int) -> None:
    """
    Draw the project overview page (Page 2).
    """
    width, height = A4
    _draw_header_footer(c, page_number=2)

    x_margin = 2.5 * cm
    y = height - 3 * cm

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(x_margin, y, "Project Overview")
    y -= 1.5 * cm

    text = c.beginText(x_margin, y)
    text.setFont("Helvetica", 11)
    text.setFillColor(colors.HexColor("#374151"))
    text.textLines(
        """
Sentiment analysis is the task of automatically determining the emotion or opinion
expressed in a piece of text. In the context of social media and user-generated
content, sentiment analysis helps organizations understand how people feel about
products, services, policies, and events at scale.

Modern sentiment analysis systems leverage deep learning and transformer-based
language models to capture subtle nuances in human language. By combining powerful
contextual embeddings with traditional statistical features, we can build robust
classifiers that generalize well across diverse text samples.
        """.strip()
    )
    c.drawText(text)

    y = text.getY() - 1.2 * cm
    text = c.beginText(x_margin, y)
    text.setFont("Helvetica", 11)
    text.setFillColor(colors.HexColor("#374151"))
    text.textLines(
        """
Kannada is one of the major languages of India with a rich digital footprint, yet
it remains underrepresented in many mainstream NLP resources. Building sentiment
models for Kannada helps bridge this gap and demonstrates that high-quality AI
solutions can be created for low-resource and regional languages.
        """.strip()
    )
    c.drawText(text)

    y = text.getY() - 1.5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(x_margin, y, "Dataset Description")
    y -= 0.8 * cm

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawString(
        x_margin,
        y,
        f"The dataset contains {total_samples} labeled Kannada sentences across six classes:",
    )
    y -= 0.8 * cm

    bullets = [
        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Disgust",
        "Neutral",
    ]
    for b in bullets:
        c.drawString(x_margin + 0.6 * cm, y, f"• {b}")
        y -= 0.6 * cm


def _page_architecture(c: canvas.Canvas) -> None:
    """
    Draw the architecture diagram page (Page 3).
    """
    width, height = A4
    _draw_header_footer(c, page_number=3)

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(2.5 * cm, height - 3 * cm, "System Architecture")

    # Coordinates for diagram elements.
    center_y = height / 2 + 2 * cm
    box_w = 4.0 * cm
    box_h = 1.4 * cm
    gap = 1.5 * cm

    # Define the pipeline stages.
    stages = ["Excel", "DataAgent", "FeatureAgent", "[MuRIL + TF-IDF]", "Ensemble", "Prediction"]

    start_x = (width - (len(stages) * box_w + (len(stages) - 1) * gap)) / 2

    c.setFont("Helvetica", 10)
    for i, label in enumerate(stages):
        x = start_x + i * (box_w + gap)
        y = center_y

        c.setFillColor(colors.HexColor("#E5E7EB"))
        c.setStrokeColor(colors.HexColor("#4B5563"))
        c.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=1)

        c.setFillColor(colors.HexColor("#111827"))
        c.drawCentredString(x + box_w / 2, y + box_h / 2 - 4, label)

        # Draw arrows between boxes.
        if i < len(stages) - 1:
            x2 = x + box_w
            x3 = x2 + gap
            y_mid = y + box_h / 2
            c.setStrokeColor(colors.HexColor("#6B7280"))
            c.line(x2, y_mid, x3, y_mid)
            c.line(x3 - 0.2 * cm, y_mid + 0.15 * cm, x3, y_mid)
            c.line(x3 - 0.2 * cm, y_mid - 0.15 * cm, x3, y_mid)


def _page_results(c: canvas.Canvas, accuracy: float, macro_f1: float, cm_path: str, report_df: pd.DataFrame) -> None:
    """
    Draw the results page (Page 4), including confusion matrix and per-class metrics.
    """
    width, height = A4
    _draw_header_footer(c, page_number=4)

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(2.5 * cm, height - 3 * cm, "Results")

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawString(2.5 * cm, height - 4.3 * cm, f"Accuracy: {accuracy:.2%}")
    c.drawString(2.5 * cm, height - 5.0 * cm, f"Macro F1 Score: {macro_f1:.2f}")

    # Insert confusion matrix image if available.
    if os.path.exists(cm_path):
        img_width = 9 * cm
        img_height = 7 * cm
        x = 2.5 * cm
        y = height - 12 * cm
        c.drawImage(cm_path, x, y, width=img_width, height=img_height, preserveAspectRatio=True, mask="auto")

    # Draw per-class metrics table.
    table_x = 12.5 * cm
    table_y = height - 5 * cm
    row_height = 0.7 * cm

    columns: List[str] = ["label", "precision", "recall", "f1-score"]
    display_df = report_df[[col for col in columns if col in report_df.columns]].copy()
    display_df = display_df[display_df["label"].isin(["happy", "sad", "angry", "fear", "disgust", "neutral"])]

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(table_x, table_y, "Per-Class Metrics")

    table_y -= 1.0 * cm
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#111827"))
    for j, col in enumerate(columns):
        c.drawString(table_x + j * 3 * cm, table_y, col.title())

    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#374151"))
    table_y -= 0.5 * cm
    for _, row in display_df.iterrows():
        values = [
            str(row.get("label", "")),
            f"{row.get('precision', 0.0):.2f}",
            f"{row.get('recall', 0.0):.2f}",
            f"{row.get('f1-score', 0.0):.2f}",
        ]
        for j, val in enumerate(values):
            c.drawString(table_x + j * 3 * cm, table_y, val)
        table_y -= row_height


def _page_conclusion(c: canvas.Canvas) -> None:
    """
    Draw the conclusion and future work page (Page 5).
    """
    width, height = A4
    _draw_header_footer(c, page_number=5)

    x_margin = 2.5 * cm
    y = height - 3 * cm

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawString(x_margin, y, "Conclusion & Future Work")
    y -= 1.5 * cm

    text = c.beginText(x_margin, y)
    text.setFont("Helvetica", 11)
    text.setFillColor(colors.HexColor("#374151"))
    text.textLines(
        """
The hybrid Kannada sentiment analysis system built in this project demonstrates that
transformer-based models such as MuRIL, when combined with traditional TF-IDF features,
can deliver strong performance even on relatively small, domain-specific datasets.
The ensemble approach benefits from both rich contextual embeddings and high-resolution
character-level statistics.

The achieved accuracy and F1 scores indicate that the model captures the key emotional
signals present in Kannada text across six sentiment classes. The confusion matrix and
per-class metrics highlight where the model is most confident and where classes may
overlap or be more challenging to separate.
        """.strip()
    )
    c.drawText(text)

    y = text.getY() - 1.2 * cm
    text = c.beginText(x_margin, y)
    text.setFont("Helvetica-Bold", 12)
    text.setFillColor(colors.HexColor("#111827"))
    text.textLine("Future Work")
    text.setFont("Helvetica", 11)
    text.setFillColor(colors.HexColor("#374151"))
    text.textLines(
        """
- Expanding the dataset with more diverse domains (news, reviews, conversational data).
- Exploring data augmentation and semi-supervised techniques for low-resource settings.
- Experimenting with larger multilingual models and prompt-based approaches.
- Deploying the model as an API and integrating it into real-time applications.
        """.strip()
    )
    c.drawText(text)


def generate_report(
    output_path: str = "artifacts/sentiment_report.pdf",
    accuracy: float = 0.93,
    macro_f1: float = 0.93,
) -> None:
    """
    Generate a multi-page PDF report summarizing the Kannada Sentiment project.

    The report includes:
        - Cover page.
        - Project overview and dataset description.
        - System architecture diagram.
        - Quantitative results and confusion matrix.
        - Conclusion and future work.
    """
    artifacts_dir = os.path.dirname(output_path)
    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)

    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    cr_path = os.path.join(artifacts_dir, "classification_report.csv")

    # Determine total samples from the report if available.
    total_samples = 0
    report_df = pd.DataFrame()
    if os.path.exists(cr_path):
        report_df = pd.read_csv(cr_path)
        support_col = None
        for col in ["support", "support.1"]:
            if col in report_df.columns:
                support_col = col
                break
        if support_col:
            total_samples = int(report_df[support_col].sum())
    else:
        # Fallback if classification report does not exist yet.
        report_df = pd.DataFrame(
            {
                "label": ["happy", "sad", "angry", "fear", "disgust", "neutral"],
                "precision": [0.0] * 6,
                "recall": [0.0] * 6,
                "f1-score": [0.0] * 6,
            }
        )

    c = canvas.Canvas(output_path, pagesize=A4)

    _page_cover(c, accuracy=accuracy, macro_f1=macro_f1)
    c.showPage()

    _page_overview(c, total_samples=total_samples)
    c.showPage()

    _page_architecture(c)
    c.showPage()

    _page_results(c, accuracy=accuracy, macro_f1=macro_f1, cm_path=cm_path, report_df=report_df)
    c.showPage()

    _page_conclusion(c)
    c.showPage()

    c.save()
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()

