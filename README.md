# MNIST Parity–KNN–PCA–XGBoost Pipeline

**Repository:** IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

**Author:** Kiran Kumar P, DA24B008, IIT Madras (2025–26)

---

## 📌 Project purpose

This repository implements a hybrid MNIST digit-classification pipeline used for the end-semester project. The pipeline combines several classical ML building blocks that I implemented from scratch:

* PCA for dimensionality reduction
* K-Nearest Neighbours (KNN) used in multiple ensemble roles (one-vs-rest, parity-specific OVO pairs, even/odd binary)
* XGBoost-style gradient-boosted trees (binary classifier) implemented from first principles for parity prediction
* Support Vector Machine (simple SGD-style SVM)
* Random Forest (with a custom decision tree implementation)
* Softmax (multiclass logistic) regression
* KMeans clustering utility

The overall **parity-aware pipeline** follows this strategy:

1. Reduce input images using PCA.
2. Train 10 one-vs-rest KNNs (one per digit) on PCA features.
3. Train a KNN binary classifier for even-vs-odd on PCA features.
4. Train an XGBoost binary classifier on a separate PCA projection to predict parity (even/odd) robustly.
5. For parity disagreements between the OVR prediction and the parity predictor, refine digit prediction using pairwise (OVO) KNNs restricted to even or odd digit groups.

This hybrid approach was chosen to explore how parity information (a very cheap bit of structure) can be used to correct primary multiclass mistakes.

---

## 📁 Repository structure (recommended)

```
README.md                        # this file
algorithms.py                    # implementations of PCA, KNN, XGBoostClassifier, SVM, RandomForest, etc.
main.py                          # data loading, training and full_parity pipeline runner (script shown in assignment)
MNIST_train.csv                  # training data (features, label, even)
MNIST_validation.csv             # validation data (optional)
MNIST_test.csv                   # test split for final evaluation
requirements.txt                 # pinned python dependencies
results/                         # plots, confusion matrices, experiment notes

```

> **Note for TAs:** The evaluation will be performed using the `.py` files only. Make sure `algorithms.py` and `main.py` are present and runnable.

---

## 🧰 Dependencies

The code uses the following packages (minimum required):

* Python 3.9+
* numpy
* pandas
* scikit-learn (for metrics and helper utilities)

Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell)
pip install -r requirements.txt
```
---

## ▶️ How to run (command-line — recommended for grading)

All experiments are runnable from the command line. The main driver script shown in the assignment snippet is `main.py`.

### Example — run full parity pipeline and print F1 on test set

```bash
python main.py
```

This will:

* read `MNIST_train.csv`, `MNIST_validation.csv`, `MNIST_test.csv` (paths can be edited inside the script),
* fit PCA projections, KNNs, XGBoost parity model and OVO/KNN refiners,
* run the full-parity prediction pipeline on the test set,
* print the final weighted F1-score.

### (Optional) Running with different hyperparameters

You can modify hyperparameters directly in `main.py` or extend it to accept command-line flags (argparse). Key knobs to try:

* `PCA(n_components=...)` — PCA dimensions (used separately for KNN and XGBoost parts)
* `KNN(k=1)` — K for nearest neighbours
* `XGBoostClassifier(n_estimators, learning_rate, max_depth, lambda_l2, gamma)` — parity classifier hyperparams
* `RandomForestClassifier(n_estimators=...)` — if you want to swap classifiers

---

## ✅ Output and evaluation

The driver prints the final weighted F1-score to the console. For additional analysis you can add:

* Confusion matrix plots using `sklearn.metrics.ConfusionMatrixDisplay`
* Per-class precision/recall using `sklearn.metrics.classification_report`
* Agreement statistics between the XGBoost parity predictor and ground-truth parity

Example code snippet to print a classification report:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## Reproducibility tips

* Set `np.random.seed(...)` near the top of scripts for consistent bootstrap sampling and centroid initialization.
* Save PCA components (`pca.components`) and model objects (via `pickle`) if you want to avoid re-training every run.
* Keep train/val/test splits fixed and record the CSV filenames in `README` or an experiment log.

---

## 🧾 Authors

**Kiran Kumar P, DA24B008**, IIT Madras (2025–26)

---

## Best practices (grading checklist)

* Keep commits small and descriptive.
* Make sure `.py` files contain clear docstrings and comments (TAs evaluate `.py` files).
* Do not collaborate or share code with other students for this assignment — academic honesty policy.
* Modularize code: keep algorithm implementations in `algorithms.py` and the experiment driver in `main.py`.

---

