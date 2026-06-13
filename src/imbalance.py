# experiments/imbalance.py

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.utils import resample

from src.data_loader import load_datasets
from src.preprocess import preprocess
from src.utils import evaluate, plot_confusion_matrix


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


cat_order = ["normal", "DoS", "Probe", "R2L", "U2R"]

def oversample_minority(X_train, y_train, target_count=5000):
    y_train = pd.Series(y_train).reset_index(drop=True)
    sampled_indices = []
    
    for label in y_train.unique():
        label_indices = y_train[y_train == label].index.to_numpy()

        if label in ["R2L", "U2R"]:
            selected = resample(label_indices, replace=True, n_samples=target_count, random_state=42)
        else:
            selected = label_indices
        sampled_indices.extend(selected)

    sampled_indices = np.array(sampled_indices)
    np.random.seed(42)
    np.random.shuffle(sampled_indices)

    X_over = X_train[sampled_indices]
    y_over = y_train.iloc[sampled_indices].reset_index(drop=True)

    return X_over, y_over


def undersample_majority(X_train, y_train, max_count=5000):
    y_train = pd.Series(y_train).reset_index(drop=True)
    sampled_indices = []

    for label in y_train.unique():
        label_indices = y_train[y_train == label].index.to_numpy()
        if label in ["normal", "DoS", "Probe"] and len(label_indices) > max_count:
            selected = resample(label_indices, replace=False, n_samples=max_count, random_state=42)
        else:
            selected = label_indices

        sampled_indices.extend(selected)
    sampled_indices = np.array(sampled_indices)
    np.random.seed(42)
    np.random.shuffle(sampled_indices)
    X_under = X_train[sampled_indices]
    y_under = y_train.iloc[sampled_indices].reset_index(drop=True)
    
    return X_under, y_under

def train_and_evaluate(model, X_train, y_train, X_test, y_test, setting_name):
    print(f"\n--- {setting_name} ---")
    model.fit(X_train, y_train)
    test_result = evaluate(model, X_test, y_test, model_name=setting_name, split_name="test", experiment="multiclass")
    y_pred = model.predict(X_test)
    recalls = recall_score(y_test, y_pred,labels=cat_order, average=None, zero_division=0)
    
    for i, recall in enumerate(recalls):
        test_result[f"recall_{cat_order[i]}"] = recall
    print(classification_report(y_test, y_pred, labels=cat_order, zero_division=0))
    
    safe_name = setting_name.lower().replace(" ", "_").replace("=", "")
    plot_confusion_matrix(model, X_test, y_test, labels=cat_order, save_path=f"{RESULT_DIR}/cm_imbalance_{safe_name}.png")
    return test_result



train_df, test_df = load_datasets("./data/NSL-KDD/KDDTrain+.txt", "./data/NSL-KDD/KDDTest+.txt")
X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, y_train_cat, y_val_cat, y_test_cat = preprocess(train_df, test_df)

results = []
rf_plain = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
results.append(train_and_evaluate(rf_plain, X_train, y_train_cat, X_test, y_test_cat, setting_name="plain"))

rf_balanced = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1)
results.append(train_and_evaluate(rf_balanced, X_train, y_train_cat, X_test, y_test_cat, setting_name="class_weight_balanced"))

X_over, y_over = oversample_minority(X_train, y_train_cat, target_count=5000)

rf_oversampling = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
results.append(train_and_evaluate(rf_oversampling, X_over, y_over, X_test, y_test_cat, setting_name="oversampling"))

X_under, y_under = undersample_majority(X_train, y_train_cat, max_count=5000)

rf_undersampling = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

results.append(train_and_evaluate(rf_undersampling, X_under, y_under, X_test, y_test_cat, setting_name="undersampling"))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULT_DIR}/imbalance_comparison.csv", index=False)

print(results_df.to_string(index=False))
print(f"Saved to {RESULT_DIR}/imbalance_comparison.csv")