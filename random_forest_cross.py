import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score


# =========================
#  DODATKOWE METRYKI (BIN)
# =========================
def binary_metrics_from_cm(y_true, y_pred, pos_label="3-4"):
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    if len(classes) != 2:
        raise ValueError(f"To nie jest problem binarny: classes={classes}")
    if pos_label not in classes:
        raise ValueError(f"pos_label={pos_label} nie występuje w danych: classes={classes}")

    neg_label = classes[classes != pos_label][0]
    labels = [neg_label, pos_label]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0,
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, average="binary"),
        "recall": recall,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }


# =========================
#  PARAM GRID (RF)
# =========================
param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 4, 5, 7, 10, 15 ],
    "min_samples_leaf": [1, 2, 5, 6, 8, 10],
    "max_features": ["sqrt"],
    "bootstrap": [True],
    "class_weight": ["balanced", None]
}

df = pd.read_csv("Data/Modified/Full/Final_reduced/2-class/Scaled/patients.csv")

target_col = "FSCORE"
#feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]"]
feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "RITIS", "NAFLD"]

df[target_col] = df[target_col].astype(str).str.strip()
allowed = {"1-2", "3-4"}
df = df[df[target_col].isin(allowed)].copy()

X = df[feature_cols].copy()
y = df[target_col].copy()

POS_LABEL = "3-4"

# ---- Nested CV ----
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

fold_results = []
best_params_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",   # selekcja hiperparametrów
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_params_per_fold.append(grid.best_params_)

    y_pred = grid.predict(X_test)

    m = binary_metrics_from_cm(y_test, y_pred, pos_label=POS_LABEL)
    m["fold"] = fold_idx
    fold_results.append(m)

res = pd.DataFrame(fold_results).set_index("fold")

print("Nested CV results (mean ± std):")
for col in ["acc", "f1", "recall", "specificity", "ppv", "npv"]:
    print(f"  {col:12s}: {res[col].mean():.4f} ± {res[col].std():.4f}")

print("\nBest params per outer fold:")
for i, bp in enumerate(best_params_per_fold, 1):
    print(f"  Fold {i}:", bp)

print("\nPer-fold counts (tn, fp, fn, tp):")
print(res[["tn", "fp", "fn", "tp"]])

# ---- FINAL (GridSearch na całości) ----
cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_final = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_final,
    n_jobs=-1,
    refit=True
)
grid_final.fit(X, y)

print("\nFINAL best params:", grid_final.best_params_)
print("FINAL CV best score:", grid_final.best_score_)

final_model = grid_final.best_estimator_

# (opcjonalnie) metryki in-sample – orientacyjnie
y_pred_all = final_model.predict(X)
final_m = binary_metrics_from_cm(y, y_pred_all, pos_label=POS_LABEL)
print("\nFINAL model metrics on ALL data (in-sample, może być zawyżone):")
for col in ["acc", "f1", "recall", "specificity", "ppv", "npv"]:
    print(f"  {col:12s}: {final_m[col]:.4f}")
print("  Counts (tn, fp, fn, tp):", final_m["tn"], final_m["fp"], final_m["fn"], final_m["tp"])
