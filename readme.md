# NSL-KDD Intrusion Detection Classification

This project trains and evaluates machine-learning models on the NSL-KDD intrusion detection dataset. It includes binary classification, multiclass attack-family classification, class-imbalance experiments, and feature-ablation analysis.

## Project Structure

```text
.
├── data/
│   └── NSL-KDD/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
├── results/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_binary.py
│   ├── multiclass.py
│   ├── imbalance.py
│   ├── ablation.py
│   └── utils.py
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Running the Project

Run all commands from the project root directory.

### 1. Generate dataset distribution plots

```bash
python -m src.data_loader
```

This saves class-distribution plots in the `results/` folder.

### 2. Run binary classification experiments

```bash
python -m src.train_binary
```

This trains and evaluates binary normal-vs-attack models and saves the main binary results and plots to `results/`.

Main outputs:

```text
results/binary_results.csv
results/binary_model_comparison.png
results/cm_decision_tree_depth=5.png
results/cm_decision_tree_depth=10.png
results/cm_decision_tree_depth=20.png
results/cm_naive_bayes.png
results/cm_boosting.png
results/cm_random_forest.png
```

### 3. Run multiclass classification experiments

```bash
python -m src.multiclass
```

This trains and evaluates models for five attack categories:

```text
normal, DoS, Probe, R2L, U2R
```

Main outputs:

```text
results/multiclass_results.csv
results/cm_multiclass_random_forest_multiclass.png
results/cm_multiclass_boosting_multiclass.png
```

### 4. Run class-imbalance experiments

```bash
python -m src.imbalance
```

This compares several imbalance-handling settings:

```text
plain
class_weight_balanced
oversampling
undersampling
```

Main outputs:

```text
results/imbalance_comparison.csv
results/cm_imbalance_plain.png
results/cm_imbalance_class_weight_balanced.png
results/cm_imbalance_oversampling.png
results/cm_imbalance_undersampling.png
```

### 5. Run feature-ablation experiments

```bash
python -m src.ablation
```

This evaluates different feature groups:

```text
basic
content
traffic
all_features
```

Main outputs:

```text
results/ablation.csv
results/ablation.png
```

## Full Experiment Run

To run the main experiments one after another:

```bash
python -m src.data_loader
python -m src.train_binary
python -m src.multiclass
python -m src.imbalance
python -m src.ablation
```
