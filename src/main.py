import os
import pandas as pd

from preprocessing import load_data
from feature_engineering import create_labels, create_features
from model import train_model, feature_importance

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score


def main():
    # =========================
    # 1. PATH SETUP
    # =========================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")

    # =========================
    # 2. LOAD + PREPROCESS (MULTI-BATTERY)
    # =========================
    battery_files = ["B0005.mat", "B0006.mat"]  # add more if available

    dfs = []

    for file in battery_files:
        file_path = os.path.join(data_dir, file)

        temp_df = load_data(file_path)

        # Add battery ID
        battery_id = file.replace(".mat", "")
        temp_df["battery_id"] = battery_id

        # Feature engineering + labeling
        temp_df = create_features(temp_df)
        temp_df = create_labels(temp_df)

        dfs.append(temp_df)

    # Combine all batteries into one dataset
    df = pd.concat(dfs, ignore_index=True)
    # =========================
    # 3. BASELINE MODEL (your current setup)
    # =========================
    model, features, baseline_acc = train_model(df)

    print(f"Random Split Accuracy: {baseline_acc:.4f}")

    # =========================
    # 4. PREPARE DATA FOR VALIDATION TESTS
    # =========================
    X = df[features]
    y = df["label"]   # ⚠️ make sure this matches your label column
    groups = df["battery_id"]  # ⚠️ MUST exist

    # =========================
    # 5. GROUP SPLIT (realistic generalization)
    # =========================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in gss.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_g, _, _ = train_model(pd.concat([X_train, y_train], axis=1))

    y_pred = model_g.predict(X_test)
    group_acc = accuracy_score(y_test, y_pred)

    print(f"Group Split Accuracy: {group_acc:.4f}")

    # =========================
    # 6. NO-CAPACITY TEST
    # =========================
    if "capacity" in X.columns:
        X_no_cap = X.drop(columns=["capacity"])

        for train_idx, test_idx in gss.split(X_no_cap, y, groups):
            X_train_nc = X_no_cap.iloc[train_idx]
            X_test_nc = X_no_cap.iloc[test_idx]
            y_train_nc = y.iloc[train_idx]
            y_test_nc = y.iloc[test_idx]

        model_nc, _, _ = train_model(pd.concat([X_train_nc, y_train_nc], axis=1))

        y_pred_nc = model_nc.predict(X_test_nc)
        no_cap_acc = accuracy_score(y_test_nc, y_pred_nc)

        print(f"No Capacity Accuracy: {no_cap_acc:.4f}")
    else:
        no_cap_acc = None
        print("Capacity column not found — skipping no-capacity test")

    # =========================
    # 7. FEATURE IMPORTANCE
    # =========================
    feature_importance(model, features)

    # =========================
    # 8. SAVE RESULTS
    # =========================
    results_path = os.path.join(BASE_DIR, "results", "metrics.txt")

    with open(results_path, "w") as f:
        f.write(f"Random Split Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Group Split Accuracy: {group_acc:.4f}\n")
        if no_cap_acc is not None:
            f.write(f"No Capacity Accuracy: {no_cap_acc:.4f}\n")


if __name__ == "__main__":
    main()