
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "breast_cancer.csv"


def create_dataset() -> Path:

    dataset = load_breast_cancer(as_frame=True)

    frame = dataset.data.copy()

    # Target: 0 = malignant, 1 = benign
    frame["target"] = dataset.target.astype("int64")

    # Add unique patient IDs
    frame.insert(
        0,
        "patient_id",
        [f"patient_{i:04d}" for i in range(len(frame))]
    )

    # Create output folder
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset as CSV
    frame.to_csv(OUT, index=False)

    print(f"Saved: {OUT}")
    print(f"Shape: {frame.shape}")

    print(
        frame["target"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return OUT


if __name__ == "__main__":
    create_dataset()
