
import pandas as pd
import yaml

from pathlib import Path
from sklearn.model_selection import train_test_split


# Project root directory
ROOT = Path(__file__).resolve().parents[2]


def load_parameters():
    """Load parameters from params.yaml."""
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_features():
    params = load_parameters()

    # Read configuration
    input_path = ROOT / params["data"]["input_path"]
    output_dir = ROOT / params["data"]["output_dir"]

    target_col = params["features"]["target_column"]
    id_col = params["features"]["id_column"]

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # Load raw dataset
    df = pd.read_csv(input_path)

    # Validate required columns
    if target_col not in df.columns or id_col not in df.columns:
        raise ValueError("Required target or ID column is missing.")

    # Separate features and target
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Combine features and target
    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed datasets
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Feature preparation completed!")
    print("Original dataset:", df.shape)
    print("Training dataset:", train_df.shape)
    print("Testing dataset:", test_df.shape)
    print("Saved:", output_dir / "train.csv")
    print("Saved:", output_dir / "test.csv")


if __name__ == "__main__":
    build_features()
