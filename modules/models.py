import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def train_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    name="Model",
    feature_name="Feature",
    label_names=None,
    verbose=True
):
    """
    Train và evaluate một model.

    Output:
        result: dict chứa accuracy, precision, recall, f1, thời gian train
        report: classification_report dạng dict
        y_pred: nhãn dự đoán
    """
    start_time = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    result = {
        "model": name,
        "feature": feature_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_time_sec": train_time
    }

    if verbose:
        print(f"--- {name} + {feature_name} Results ---")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro-F1     : {macro_f1:.4f}")
        print(f"Weighted-F1  : {weighted_f1:.4f}")
        print(f"Train time   : {train_time:.2f} sec")

    return result, report, y_pred


def results_to_dataframe(results):
    """
    Chuyển list kết quả thành DataFrame và sort theo accuracy.
    """
    df = pd.DataFrame(results)

    if "accuracy" in df.columns:
        df = df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    return df
