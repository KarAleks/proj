from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

random_state = 42

MODELS = {"Decision Tree depth=5": DecisionTreeClassifier(max_depth=5,random_state=random_state, class_weight="balanced"),
          "Decision Tree depth=10": DecisionTreeClassifier(max_depth=10, random_state=random_state, class_weight="balanced"),
          "Decision Tree depth=20": DecisionTreeClassifier(max_depth=20, random_state=random_state, class_weight="balanced"),
          "Boosting": HistGradientBoostingClassifier(random_state=random_state), "Naive Bayes": GaussianNB(),}

# def evaluate(y_true, y_pred, y_proba=None, experiment="binary"):
#     if experiment == "binary":
#         average = "binary"
#     else:
#         average = "macro"

#     acc = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average=average)
#     recall = recall_score(y_true, y_pred, average=average)
#     f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
#     results = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

#     if y_proba is not None:
#             if experiment == "binary":
#                 results["auc"] = roc_auc_score(y_true, y_proba[:, 1])
#             else:
#                 results["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
#     return results
def evaluate(model, X, y, model_name, split_name, experiment="binary"):
    y_pred = model.predict(X)
    
    if experiment == "binary":
        average = "binary"
    else:
        average = "macro"

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=average, zero_division=0)
    recall = recall_score(y, y_pred, average=average, zero_division=0)
    f1 = f1_score(y, y_pred, average=average, zero_division=0)

    results = {"model": model_name, "split": split_name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": None}

    if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            if experiment == "binary":
                results["auc"] = roc_auc_score(y, y_proba[:, 1])
            else:
                results["auc"] = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
                
    print(f"\n{model_name} — {split_name}")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")

    if results["auc"] is not None:
        print(f"auc:       {results['auc']:.4f}")

    return results

def plot_confusion_matrix(model, X_test, y_test, labels, save_path=None):
    _ , ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels, xticks_rotation=45, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_model_comparision(results_df, metrics, save_path=None):
    x = np.arange(len(metrics))
    width = 0.8 / len(results_df)
    _ , ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax.bar(x + i * width, values, width=width, label=row["model"])

    ax.set_xticks(x + width * (len(results_df) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, top_k=15, save_path=None):
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
    names = []
    values = []
    for p in pairs:
        names.append(p[0])
        values.append(p[1])
    plt.figure(figsize=(8, 5))
    plt.barh(names, values)
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()