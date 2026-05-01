import os
from preprocessing import load_data
from feature_engineering import create_labels, create_features
from model import train_model, feature_importance


def main():
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build correct path
    file_path = os.path.join(BASE_DIR, "data", "B0005.mat")

    # Load data
    df = load_data(file_path)

    # Feature engineering
    df = create_features(df)

    # Labeling
    df = create_labels(df)

    # Train model
    model, features, acc = train_model(df)

    # Feature importance
    feature_importance(model, features)

    # Save results
    with open(os.path.join(BASE_DIR, "results", "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()