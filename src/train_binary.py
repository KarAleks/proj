import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_datasets
from src.preprocess import preprocess
from src.utils import evaluate, plot_confusion_matrix, plot_model_comparision, MODELS

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def grid_search_rf(X_train, y_train, X_val, y_val):
    best_f1 = 0
    best_model = None
    best_params = None
    for n_estimators in [50, 100, 200, 300]:
        for max_depth in [5, 10, 20, None]:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                           random_state=42, class_weight="balanced", n_jobs=-1)
            model.fit(X_train, y_train)
            val_result = evaluate(model, X_val, y_val, model_name="Random Forest", split_name="val")
            val_f1 = val_result["f1"]
            print(f"Random Forest: n_estimators={n_estimators}, max_depth={max_depth},val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = model
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    print(f"\nBest Random Forest params: {best_params}")
    print(f"Best validation F1: {best_f1:.4f}")
    return best_model, best_params

train_df, test_df = load_datasets("./data/NSL-KDD/KDDTrain+.txt", "./data/NSL-KDD/KDDTest+.txt")
X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, y_train_cat, y_val_cat, y_test_cat = preprocess(train_df, test_df)
results = []

for model_name, model in MODELS.items():
    print(f"Training {model_name}...\n")
    model.fit(X_train, y_train_bin)
    val_result = evaluate(model, X_val, y_val_bin, model_name=model_name, split_name="val")
    test_result = evaluate(model, X_test, y_test_bin, model_name=model_name, split_name="test")
    results.append(val_result)
    results.append(test_result)
    plot_confusion_matrix(model, X_test, y_test_bin, labels=["normal", "attack"], save_path=f"{RESULT_DIR}/cm_{model_name.lower().replace(' ', '_')}.png")

print("\nTuning Random Forest...")
best_rf, best_params = grid_search_rf(X_train, y_train_bin, X_val, y_val_bin)

results.append(evaluate(best_rf, X_val, y_val_bin, model_name="Random Forest", split_name="val"))
results.append(evaluate(best_rf, X_test, y_test_bin, model_name="Random Forest", split_name="test"))

plot_confusion_matrix(best_rf, X_test, y_test_bin, labels=["normal", "attack"], save_path=f"{RESULT_DIR}/cm_random_forest.png")
results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULT_DIR}/binary_results.csv", index=False)
test_results = results_df[results_df["split"] == "test"]
plot_model_comparision(test_results, metrics=["accuracy","precision", "recall", "f1"], save_path=f"{RESULT_DIR}/binary_model_comparison.png")