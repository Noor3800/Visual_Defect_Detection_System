import json
import shutil
from pathlib import Path
import pandas as pd  # type: ignore


def get_defect_type(path):
    """
    Extract defect type from the image path.

    Examples:
        .../test/good/001.png
            -> normal

        .../test/broken_large/001.png
            -> broken_large

        .../test/broken_small/001.png
            -> broken_small

        .../test/contamination/001.png
            -> contamination
    """
    path = Path(path)

    if "good" in path.parts:
        return "normal"

    return path.parent.name

# Save images belonging to FP / FN

def save_error_images(df, output_dir):
    """
    Save false-positive and false-negative images.
    """

    output_dir = Path(output_dir)

    fp_dir = output_dir / "false_positives"
    fn_dir = output_dir / "false_negatives"

    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    # False Positives:
    # Actual = normal
    # Predicted = defective
    fp = df[
        (df["label"] == "normal")
        & (df["pred"] == "defective")
    ]

    # False Negatives:
    # Actual = defective
    # Predicted = normal
    fn = df[
        (df["label"] == "defective")
        & (df["pred"] == "normal")
    ]

    # Copy FP images
    for _, row in fp.iterrows():

        source = Path(row["path"])

        if source.exists():

            confidence = row.get("prob_defective", 0)

            destination = (
                fp_dir
                / f"p{confidence:.2f}_{source.name}"
            )

            shutil.copy2(source, destination)

    # Copy FN images
    for _, row in fn.iterrows():

        source = Path(row["path"])

        if source.exists():

            confidence = row.get("prob_defective", 0)

            destination = (
                fn_dir
                / f"p{confidence:.2f}_{source.name}"
            )

            shutil.copy2(source, destination)

    return fp, fn


# Calculate recall for each defect type
def calculate_defect_recall(df):
    """
    Calculate recall separately for every MVTec defect type.
    """

    defective = df[df["label"] == "defective"].copy()

    if defective.empty:
        return {}

    defective["caught"] = (
        defective["pred"] == "defective"
    )

    recall_by_type = (
        defective
        .groupby("defect_type")["caught"]
        .agg(["sum", "count"])
    )

    recall_by_type["recall"] = (
        recall_by_type["sum"]
        / recall_by_type["count"]
    )

    return (
        recall_by_type["recall"]
        .round(3)
        .to_dict()
    )


def main():

    # Project root
    project_root = Path(__file__).resolve().parents[1]

    # Results directory
    results_dir = project_root / "results"

    # Evaluation predictions
    predictions_file = (
        results_dir / "test_predictions.csv"
    )

    if not predictions_file.exists():

        raise FileNotFoundError(
            f"Could not find:\n{predictions_file}\n\n"
            "Run evaluate.py first so that "
            "test_predictions.csv is created."
        )

    print("\nLoading predictions...")
    df = pd.read_csv(predictions_file)


    required_columns = {
        "path",
        "label",
        "pred",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing columns in test_predictions.csv: "
            f"{sorted(missing)}"
        )


    df["defect_type"] = df["path"].apply(
        get_defect_type
    )


    # Find FP and FN

    fp, fn = save_error_images(
        df,
        results_dir / "errors"
    )


    recall_by_type = calculate_defect_recall(df)

    # Create summary

    summary = {
        "total_test_images": int(len(df)),
        "false_positives": int(len(fp)),
        "false_negatives": int(len(fn)),
        "false_positive_files": (
            fp["path"].tolist()
        ),
        "false_negative_files": (
            fn["path"].tolist()
        ),
        "recall_by_defect_type": recall_by_type,
    }

    # Save summary

    summary_file = (
        results_dir / "error_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )



    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)

    print(
        f"\nTotal test images: "
        f"{len(df)}"
    )

    print(
        f"False Positives: "
        f"{len(fp)}"
    )

    print(
        f"False Negatives: "
        f"{len(fn)}"
    )

    print("\nRecall by defect type:")

    if recall_by_type:

        for defect, recall in recall_by_type.items():

            print(
                f"  {defect}: "
                f"{recall:.3f}"
            )

    else:

        print("  No defective samples found.")

    print(
        f"\nError images saved to:"
        f"\n{results_dir / 'errors'}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_file}"
    )

    print("\nError analysis completed successfully.")



if __name__ == "__main__":
    main()