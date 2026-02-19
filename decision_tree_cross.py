import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

label_map = {"1-2": 0, "3-4": 1}
POS_LABEL = 1

def binary_cm_metrics(y_true, y_pred, pos_label=1):
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    if len(classes) != 2:
        raise ValueError(f"To nie jest problem binarny: classes={classes}")
    if pos_label not in classes:
        raise ValueError(f"pos_label={pos_label} nie występuje w danych: classes={classes}")

    neg_label = classes[classes != pos_label][0]
    labels = [neg_label, pos_label]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0                 # sensitivity, TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0.0            # TNR
    ppv = tp / (tp + fp) if (tp + fp) else 0.0                    # precision
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0,
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }

# ----------------- DANE -----------------
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1],
}

df = pd.read_csv("Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv")

target_col = "FSCORE"
#feature_cols = ["FIB4", "RITIS", "APRI", "NAFLD"]
feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "RITIS", "NAFLD"]

# Ujednolicenie etykiet
df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col].isin(label_map)].copy()
df["y_bin"] = df[target_col].map(label_map).astype(int)

X = df[feature_cols].copy()
y = df["y_bin"].copy()

# ---- Nested CV ----
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = DecisionTreeClassifier(random_state=42)

fold_results = []
best_params_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_params_per_fold.append(grid.best_params_)

    y_pred = grid.predict(X_test)

    m = binary_cm_metrics(y_test, y_pred, pos_label=POS_LABEL)
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

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_final = GridSearchCV( estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, scoring="f1_macro", cv=cv_final, n_jobs=-1, refit=True )
grid_final.fit(X, y)
print("FINAL best params:", grid_final.best_params_)
print("FINAL CV best score:", grid_final.best_score_)
final_model = grid_final.best_estimator_
# --- FINAL: metryki in-sample (orientacyjnie) ---
y_pred_all = final_model.predict(X)
final_metrics = binary_cm_metrics(y, y_pred_all, pos_label=POS_LABEL)

print("\nFINAL model metrics on ALL data (in-sample, może być zawyżone):")
for k in ["acc", "f1", "recall", "specificity", "ppv", "npv"]:
    print(f"  {k:12s}: {final_metrics[k]:.4f}")
print("  Confusion (tn, fp, fn, tp):",
      final_metrics["tn"], final_metrics["fp"], final_metrics["fn"], final_metrics["tp"])
