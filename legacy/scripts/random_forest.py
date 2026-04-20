from cProfile import label

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# =========================
#  DODATKOWE METRYKI (BIN)
# =========================
def binary_metrics_from_cm(y_true, y_pred, pos_label="3-4"):
    """
    Metryki dla problemu binarnego na bazie macierzy pomyłek:
    recall/sensitivity, specificity, PPV, NPV.
    Działa dla etykiet stringowych typu "1-2" / "3-4".
    """
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    if len(classes) != 2:
        raise ValueError(f"To nie jest problem binarny: classes={classes}")
    if pos_label not in classes:
        raise ValueError(f"pos_label={pos_label} nie występuje w danych: classes={classes}")

    neg_label = classes[classes != pos_label][0]
    labels = [neg_label, pos_label]  # [neg, pos]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0                 # sensitivity (TPR)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0            # TNR
    ppv = tp / (tp + fp) if (tp + fp) else 0.0                    # precision
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "pos_label": pos_label,
        "neg_label": neg_label,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }


# =========================
#  PARAM GRID (RF)
# =========================
param_grid = {
    "n_estimators": [100],
    "max_depth": [None, 6],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
    "bootstrap": [True],
    "class_weight": [None, "balanced"]
}


train_df = pd.read_csv("Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv")
test_df  = pd.read_csv("Data/Test-group/control-imp-2-class.csv")

target_col = "FSCORE"
feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "RITIS", "NAFLD"]

train_df[target_col] = train_df[target_col].astype(str).str.strip()
test_df[target_col]  = test_df[target_col].astype(str).str.strip()

allowed = {"1-2", "3-4"}
train_df = train_df[train_df[target_col].isin(allowed)].copy()
test_df  = test_df[test_df[target_col].isin(allowed)].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test  = test_df[feature_cols].copy()
y_test  = test_df[target_col].copy()

# =========================
#  GRIDSEARCH + CV
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=0,
    refit=True
)

grid.fit(X_train, y_train)

print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik CV (średni):", grid.best_score_)

best_model = grid.best_estimator_

# =========================
#  TEST
# =========================
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
fig, ax = plt.subplots(figsize=(6, 5))
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['F1-F2', 'F3-F4'])
cm_disp.plot(cmap='Blues')
ax.set_xlabel("Klasa przewidziana")   # zamiast Predicted label
ax.set_ylabel("Klasa rzeczywista")    # zamiast True label

ax.set_title("Macierz pomyłek - Drzewo decyzyjne, FSCORE: 1-2 vs. 3-4")
plt.savefig('forest_confusion_matrix_v2.png')
plt.close()

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

m = binary_metrics_from_cm(y_test, y_pred, pos_label="3-4")
print("\nDodatkowe metryki (pozytywna klasa = '3-4'):")
print(f"  Recall / Sensitivity (TPR): {m['recall']:.4f}")
print(f"  Specificity (TNR):          {m['specificity']:.4f}")
print(f"  PPV (Precision):            {m['ppv']:.4f}")
print(f"  NPV:                        {m['npv']:.4f}")
print(f"  Counts (tn, fp, fn, tp):    {m['tn']}, {m['fp']}, {m['fn']}, {m['tp']}")

# =========================
#  (OPCJONALNIE) FEATURE IMPORTANCES
# =========================
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)