from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    return model, X.columns, acc


import os
import matplotlib.pyplot as plt

def feature_importance(model, feature_names):
    importances = model.feature_importances_

    # Get project root (battery-health-ml/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create results directory if not exists
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Plot
    plt.figure()
    plt.barh(feature_names, importances)
    plt.title("Feature Importance")

    # Save correctly
    save_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to: {save_path}")
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name}: {acc}")
from sklearn.metrics import confusion_matrix

