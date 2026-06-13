# src/ablation.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_datasets
from src.preprocess import preprocess
from src.utils import evaluate


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


BASIC_FEATURES = ["duration", "protocol_type", "service", "flag", "src_bytes",
                  "dst_bytes", "land", "wrong_fragment", "urgent"]

CONTENT_FEATURES = ["hot", "num_failed_logins", "logged_in", "num_compromised",
                    "root_shell", "su_attempted", "num_root", "num_file_creations",
                    "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login"]

TRAFFIC_FEATURES = ["count", "srv_count", "serror_rate", "srv_serror_rate",
                    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                    "dst_host_srv_count", "dst_host_same_srv_rate",
                    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]


def run_ablation(train_df, test_df, feature_cols, group_name):
    
    X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, _, _, _ = preprocess(train_df, test_df, feature_cols=feature_cols)
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train_bin)
    val_result = evaluate(model, X_val, y_val_bin, model_name=group_name, split_name="val", experiment="binary")
    test_result = evaluate(model, X_test, y_test_bin, model_name=group_name, split_name="test", experiment="binary")
    val_result["n_features"] = X_train.shape[1]
    test_result["n_features"] = X_train.shape[1]
    return val_result, test_result



train_df, test_df = load_datasets("./data/NSL-KDD/KDDTrain+.txt", "./data/NSL-KDD/KDDTest+.txt")
experiments = {"basic": BASIC_FEATURES, "content": CONTENT_FEATURES,
                "traffic": TRAFFIC_FEATURES, "all_features": BASIC_FEATURES + CONTENT_FEATURES + TRAFFIC_FEATURES}
results = []

for group_name, feature_cols in experiments.items():
    print(f"\nRunning ablation for: {group_name}")
    val_result, test_result = run_ablation(train_df, test_df, feature_cols, group_name)
    results.append(val_result)
    results.append(test_result)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULT_DIR}/ablation.csv", index=False)

print("\n--- Ablation results ---")
print(results_df.to_string(index=False))
print(f"\nSaved to {RESULT_DIR}/ablation.csv")

test_results = results_df[results_df["split"] == "test"]

metrics = ["precision", "recall", "f1"]
x = np.arange(len(test_results))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))

for i, metric in enumerate(metrics):
    ax.bar(x + i * width, test_results[metric], width=width, label=metric)

ax.set_xticks(x + width)
ax.set_xticklabels(test_results["model"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Feature Ablation — Binary Classification")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/ablation.png", dpi=150)
plt.close()
print(f"Saved to {RESULT_DIR}/ablation.png")
