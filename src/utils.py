import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import json
from typing import List, Optional
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder

def resolve_root():
    here = pathlib.Path.cwd().resolve()
    for c in [here, *list(here.parents)[:3]]:
        if (c / 'data' / 'raw' / 'weather_classification_data.csv').exists():
            return c
    return here

# def make_preprocessor(
#     numeric_features: List[str],
#     categorical_features: List[str],
#     log1p_features: Optional[List[str]] = None,
# ) -> ColumnTransformer:
#     log1p_features = list(log1p_features or [])
#     num_no_log = [f for f in numeric_features if f not in log1p_features]

#     transformers = []

#     if log1p_features:
#         transformers.append(
#             (
#                 'num_log1p',
#                 SKPipeline(
#                     steps=[
#                         ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
#                         ('scaler', StandardScaler()),
#                     ]
#                 ),
#                 log1p_features,
#             )
#         )

#     if num_no_log:
#         transformers.append(('num', StandardScaler(), num_no_log))

#     if categorical_features:
#         transformers.append(
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
#         )

#     preprocessor = ColumnTransformer(transformers)
#     return preprocessor

def make_preprocessor(cat_features, num_features, log1p_features=None):
    """Tạo preprocessor gồm xử lý numeric, categorical, và log1p (nếu có)."""

    log1p_features = log1p_features or []
    remaining_num = [f for f in num_features if f not in log1p_features]

    # Numeric pipeline
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Log1p pipeline (dành cho các cột skew cao)
    log1p_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, remaining_num),
            ("log1p", log1p_pipeline, log1p_features),
            ("cat", cat_pipeline, cat_features)
        ],
        remainder="drop"
    )

    return preprocessor

def eval_on_val(model_key,
                base_estimator,
                grid,
                preprocessor,
                X_train,
                y_train,
                X_val,
                y_val,
                label_encoder: Optional[LabelEncoder] = None):
    """
    Đánh giá các cấu hình hyperparameter trên tập validation.
    Giữ nguyên logic ban đầu: cho mỗi cfg -> tạo Pipeline(prep, clf),
    set các tham số cho clf, fit trên X_train/y_train_enc, predict X_val,
    tính macro_f1 và accuracy dựa trên y_val_enc.

    Nếu y_train / y_val là nhãn không-encoded (chuỗi hoặc số),
    hàm sẽ tự tạo LabelEncoder (hoặc dùng label_encoder nếu truyền vào).

    Parameters
    ----------
    model_key : str
        Tên mô hình (ví dụ "svm", "xgb", "logreg")
    base_estimator : estimator instance
        Một instance của estimator (dùng để lấy class và default params)
    grid : list of dict
        Danh sách cấu hình: [{"config_id": "...", "params": {...}}, ...]
    preprocessor : transformer
        Preprocessing pipeline (ColumnTransformer / Pipeline)
    X_train, y_train, X_val, y_val : array-like / DataFrame / Series
        Dữ liệu thô (y có thể là chuỗi)
    label_encoder : sklearn.preprocessing.LabelEncoder or None
        Nếu truyền vào, sẽ dùng nó; nếu None, hàm sẽ tạo + fit trên y_train.

    Returns
    -------
    pd.DataFrame
        Các hàng chứa model, config_id, params_json, macro_f1_val, accuracy_val
    """
    rows = []

    # --- label encoding (nếu cần) ---
    if label_encoder is None:
        le = LabelEncoder()
        # fit on y_train (assumes y_train is array-like)
        y_train_arr = np.array(y_train)
        le.fit(y_train_arr)
    else:
        le = label_encoder

    # transform labels
    y_train_enc = le.transform(np.array(y_train))
    y_val_enc = le.transform(np.array(y_val))

    # iterate grid (same logic as của bạn)
    for cfg in grid:
        params = cfg["params"].copy()
        cfg_id = cfg["config_id"]

        # instantiate a fresh estimator of same class with same init params
        est_cls = base_estimator.__class__
        est = est_cls(**base_estimator.get_params())

        # pipeline: preprocessor + fresh estimator
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", est)])

        # set hyperparameters on the classifier step
        if params:
            pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})

        # fit on training (encoded)
        pipe.fit(X_train, y_train_enc)

        # predict on validation
        y_pred = pipe.predict(X_val)

        # compute metrics on encoded labels
        macro_f1 = f1_score(y_val_enc, y_pred, average="macro")
        acc = accuracy_score(y_val_enc, y_pred)

        rows.append({
            "model": model_key,
            "config_id": cfg_id,
            "params_json": json.dumps(params),
            "macro_f1_val": macro_f1,
            "accuracy_val": acc,
        })

    return pd.DataFrame(rows)

# def eval_on_val(model_key, base_estimator, grid):
#     rows = []
#     for cfg in grid:
#         params = cfg["params"].copy(); cfg_id = cfg["config_id"]
#         pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_estimator.__class__(**base_estimator.get_params()))])
#         pipe.set_params(**{f"clf__{k}": v for k,v in params.items()})
#         pipe.fit(X_train, y_train_enc)
#         y_pred = pipe.predict(X_val)
#         rows.append({
#             "model": model_key, "config_id": cfg_id, "params_json": json.dumps(params),
#             "macro_f1_val": f1_score(y_val_enc, y_pred, average="macro"),
#             "accuracy_val": accuracy_score(y_val_enc, y_pred),
#         })
#     return pd.DataFrame(rows)

def softmax(logits):
    # logits: (n_samples, n_classes)
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def evaluate_and_plot(model, X_test, y_test, label_encoder=None, feature_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        # numeric class names
        target_names = [str(i) for i in range(model.n_classes)]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2. else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # feature importance bar
    fi = model.get_feature_importance(feature_names=feature_names, top_n=30)
    if len(fi) > 0:
        names, vals = zip(*fi)
        plt.figure(figsize=(8, max(3, len(vals)*0.3)))
        y_pos = np.arange(len(vals))
        plt.barh(y_pos, vals[::-1])
        plt.yticks(y_pos, names[::-1])
        plt.title("Feature importance (gain, aggregated)")
        plt.xlabel("Total Gain")
        plt.tight_layout()
        plt.show()

# --- Hàm vẽ và lưu ma trận nhầm lẫn ---
def plot_and_save_confusion(cm, classes, title, path):
    """
    Vẽ ma trận nhầm lẫn và lưu ảnh.

    Parameters
    ----------
    cm : np.ndarray
        Ma trận nhầm lẫn
    classes : list of str
        Tên lớp
    title : str
        Tiêu đề biểu đồ
    path : str / pathlib.Path
        Đường dẫn lưu file
    """
    plt.figure(figsize=(5.2,4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def roc_pr_ovr(y_true_enc, y_proba, classes, prefix, out_dir):
    """
    Vẽ ROC & PR (one-vs-rest) cho nhiều lớp và lưu ảnh.

    Parameters
    ----------
    y_true_enc : np.ndarray
        Nhãn true (encoded)
    y_proba : np.ndarray
        Dự đoán probability từ classifier
    classes : list of str
        Danh sách tên lớp
    prefix : str
        Tiền tố tên mô hình
    out_dir : str / pathlib.Path
        Thư mục lưu hình
    """
    # ROC
    plt.figure(figsize=(5.2,4.2))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_enc==i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC OvR — {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_test_roc.png", dpi=150)
    plt.close()

    # PR
    plt.figure(figsize=(5.2,4.2))
    for i, cls in enumerate(classes):
        p, r, _ = precision_recall_curve(y_true_enc==i, y_proba[:, i])
        ap = average_precision_score(y_true_enc==i, y_proba[:, i])
        plt.plot(r, p, label=f"{cls} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR OvR — {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_test_pr.png", dpi=150)
    plt.close()