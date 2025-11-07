import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

def _resolve_data_path(data_path: Optional[Path] = None) -> Path:
    if data_path is not None:
        return Path(data_path)

    repo_root = Path(__file__).resolve().parents[1] 
    default = repo_root / "data" / "raw" / "weather_classification_data.csv"
    return default


def _load_and_preprocess(
    data_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):

    data_fp = _resolve_data_path(data_path)
    df = pd.read_csv(data_fp)

    X_raw = df.drop("Weather Type", axis=1)
    y_raw = df["Weather Type"]

    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_enc,
        test_size=test_size,
        stratify=y_enc,
        random_state=random_state,
    )

    cat_features = X_raw.select_dtypes(include=["object"]).columns.tolist()
    num_features = X_raw.select_dtypes(exclude=["object"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", ohe, cat_features),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, label_enc, preprocessor


class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # y ∈ {-1, 1}
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = self.lambda_param * self.w
                    self.w -= self.lr * dw
                else:
                    dw = self.lambda_param * self.w - np.dot(x_i, y[idx])
                    self.w -= self.lr * dw
                    self.b -= self.lr * y[idx]

    def project(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        linear_output = self.project(X)
        return np.sign(linear_output)


class MultiClassSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            print(f"Training classifier for class {c}...")
            y_binary = np.where(y == c, 1, -1)
            clf = LinearSVM(self.lr, self.lambda_param, self.n_iters)
            clf.fit(X, y_binary)
            self.classifiers[c] = clf

    def predict(self, X):
        scores = np.array([self.classifiers[c].project(X) for c in self.classes])
        y_idx = np.argmax(scores, axis=0)
        y_pred = self.classes[y_idx]
        return y_pred


@dataclass
class SVMResult:
    accuracy: float
    classification_report_text: str
    confusion_matrix: np.ndarray
    labels: np.ndarray
    model: MultiClassSVM


def train_svm_scratch(
    data_path: Optional[Path] = None,
    lr: float = 0.0005,
    lambda_param: float = 0.001,
    n_iters: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
    plot: bool = True,
) -> SVMResult:
    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_enc,
        preprocessor,
    ) = _load_and_preprocess(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
    )

    model = MultiClassSVM(lr=lr, lambda_param=lambda_param, n_iters=n_iters)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {acc:.4f}")

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("SVM (from scratch) Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return SVMResult(
        accuracy=acc,
        classification_report_text=report,
        confusion_matrix=cm,
        labels=label_enc.classes_,
        model=model,
    )


def main():
    result = train_svm_scratch(plot=True)
    print(f"[train_svm1] Final accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    main()