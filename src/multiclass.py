import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from src.data_loader import load_datasets
from src.preprocess import preprocess
from src.utils import evaluate, plot_confusion_matrix

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
models = {"Random Forest Multiclass": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1),
          "Boosting Multiclass": HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.05, max_leaf_nodes=31)}

train_df, test_df = load_datasets("./data/NSL-KDD/KDDTrain+.txt", "./data/NSL-KDD/KDDTest+.txt")
X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, y_train_cat, y_val_cat, y_test_cat = preprocess(train_df, test_df)

X_train_dense = X_train#.toarray()
X_val_dense = X_val#.toarray()
X_test_dense = X_test#.toarray()

sample_weight_cat = compute_sample_weight(class_weight="balanced", y=y_train_cat)
order = ["normal", "DoS", "Probe", "R2L", "U2R"]
results = []
for model_name, model in models.items():
    nm = model_name.lower().replace(" ", "_")
    if model_name == "Boosting Multiclass":
        model.fit(X_train_dense, y_train_cat, sample_weight=sample_weight_cat)
        val_result = evaluate(model, X_val_dense, y_val_cat, model_name=model_name, split_name="val", experiment="multiclass")
        test_result = evaluate(model, X_test_dense, y_test_cat, model_name=model_name, split_name="test", experiment="multiclass")
        y_pred = model.predict(X_test_dense)
        plot_confusion_matrix(model, X_test_dense, y_test_cat, labels=order, save_path=f"{RESULT_DIR}/cm_multiclass_{nm}.png")
    else:
        model.fit(X_train, y_train_cat)
        val_result = evaluate(model, X_val, y_val_cat, model_name=model_name, split_name="val", experiment="multiclass")
        test_result = evaluate(model, X_test, y_test_cat, model_name=model_name, split_name="test", experiment="multiclass")
        y_pred = model.predict(X_test)
        plot_confusion_matrix(model, X_test, y_test_cat, labels=order, save_path=f"{RESULT_DIR}/cm_multiclass_{nm}.png")
    classific_report_test = classification_report(y_test_cat, y_pred, zero_division=0)
    results.append(val_result)
    results.append(test_result)

result_df = pd.DataFrame(results)
result_df.to_csv(f"{RESULT_DIR}/multiclass_results.csv", index=False)

# model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1)
# model.fit(X_train, y_train_cat)

# sample_weight_cat = compute_sample_weight(
#     class_weight="balanced",
#     y=y_train_cat
# )

# val_result = evaluate(model, X_val, y_val_cat, model_name="Random Forest Multiclass", split_name="val", experiment="multiclass")
# test_result = evaluate(model, X_test, y_test_cat, model_name="Random Forest Multiclass", split_name="test", experiment="multiclass")
# result_df = pd.DataFrame([val_result, test_result])
# result_df.to_csv(f"{RESULT_DIR}/multiclass_results.csv", index=False)

# y_pred = model.predict(X_test)
# y_pred_val = model.predict(X_val)
# classific_report_test = classification_report(y_test_cat, y_pred, zero_division=0)
# classific_report_val = classification_report(y_val_cat, y_pred_val, zero_division=0)

# print(classific_report_test)
# print(classific_report_val)

# order = ["normal", "DoS", "Probe", "R2L", "U2R"]
# plot_confusion_matrix(model, X_test, y_test_cat, labels=order, save_path=f"{RESULT_DIR}/cm_multiclass_random_forest.png")
# plot_confusion_matrix(model, X_val, y_val_cat, labels=order, save_path=f"{RESULT_DIR}/cm_multiclass_random_forest_val.png")