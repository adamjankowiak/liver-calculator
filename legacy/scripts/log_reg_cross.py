import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures


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


# ===== 1) Wczytanie danych =====
path = "Data/Modified/Full/Final_reduced/2-class/Scaled/patients.csv"
df = pd.read_csv(path)

feature_cols = ["Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "RITIS", "NAFLD"]
#feature_cols = ["FIB4", "APRI", "RITIS", "NAFLD"]
target_col = "FSCORE"
POS = "3-4"
NEG = "1-2"

def fscore_to_labels(v):
    if pd.isna(v):
        return np.nan
    try:
        fv = float(v)
        return POS if fv >= 3 else NEG
    except Exception:
        pass
    s = str(v).strip().lower()
    s = s.replace("–", "-").replace(" ", "")
    return POS if s == "3-4" else NEG

X = df[feature_cols].copy()
y = df[target_col].apply(fscore_to_labels)

mask = ~y.isna()
X = X.loc[mask]
y = y.loc[mask].astype(str)

# (opcjonalnie, ale spójne z podejściem "allowed" z RF)
allowed = {NEG, POS}
mask2 = y.isin(allowed)
X = X.loc[mask2]
y = y.loc[mask2]

print("Rozkład klas:")
print(y.value_counts())


# ===== 2) Pipeline (imputer + scaler + LR) =====
# Grid (elasticnet jak w Twoim kodzie)
param_grid = {
    "clf__penalty": ["elasticnet"],
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "clf__class_weight": [None, "balanced"],
    "clf__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "clf__max_iter": [1000, 3000, 10000],
    "clf__solver": ["saga"],
}

# ===== 2) Pipeline (imputer + poly(deg=2) + scaler + LR) =====

poly_cols = ["FIB4", "APRI", "RITIS", "NAFLD"]              # na tych robimy wielomiany
other_cols = [c for c in feature_cols if c not in poly_cols]  # reszta (jeśli dodasz inne cechy)

preprocess = ColumnTransformer(
    transformers=[
        ("poly", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), poly_cols),

        # jeśli w przyszłości dodasz inne cechy do feature_cols, też będą imputowane
        ("other", SimpleImputer(strategy="median"), other_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # wymuś wynik gęsty (łatwiej ze scalerem)
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        random_state=42
    ))
])


# ===== 3) Nested CV =====
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
best_params_per_fold = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params_per_fold.append(grid.best_params_)

    # predykcja etykiet (spójnie z LR: próg 0.5 na prawdopodobieństwie POS)
    pos_idx = np.where(best_model.named_steps["clf"].classes_ == POS)[0][0]
    y_proba_pos = best_model.predict_proba(X_test)[:, pos_idx]
    y_pred = np.where(y_proba_pos >= 0.5, POS, NEG)

    m = binary_metrics_from_cm(y_test.values, y_pred, pos_label=POS)
    m["fold"] = fold
    fold_metrics.append(m)

# ===== 4) Raport nested CV w stylu RF =====
print("\n=== REGRESJA LOGISTYCZNA - NESTED CV -  ===\n")

res = pd.DataFrame(fold_metrics).set_index("fold")

print("Nested CV results (mean ± std):")
for k in ["acc", "f1", "recall", "specificity", "ppv", "npv"]:
    mean = res[k].mean()
    std = res[k].std(ddof=1)
    print(f"  {k:<11}: {mean:.4f} ± {std:.4f}")

print("\nBest params per outer fold:")
for i, p in enumerate(best_params_per_fold, start=1):
    print(f"  Fold {i}: {p}")

print("\nPer-fold counts (tn, fp, fn, tp):")
print(res[["tn", "fp", "fn", "tp"]])

# ===== 5) FINAL GridSearch na CAŁOŚCI danych (jak w RF) =====
grid_final = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=inner_cv,
    n_jobs=-1,
    refit=True,
    verbose=0
)
grid_final.fit(X, y)

print("\nFINAL best params:", grid_final.best_params_)
print("FINAL CV best score:", grid_final.best_score_)

# metryki FINAL na całości (in-sample)
final_model = grid_final.best_estimator_
pos_idx = np.where(final_model.named_steps["clf"].classes_ == POS)[0][0]
y_proba_pos_all = final_model.predict_proba(X)[:, pos_idx]
y_pred_all = np.where(y_proba_pos_all >= 0.5, POS, NEG)

m_all = binary_metrics_from_cm(y.values, y_pred_all, pos_label=POS)

print("\nFINAL model metrics on ALL data (in-sample, może być zawyżone):")
for k in ["acc", "f1", "recall", "specificity", "ppv", "npv"]:
    print(f"  {k:<11}: {m_all[k]:.4f}")
print(f"  Counts (tn, fp, fn, tp): {m_all['tn']} {m_all['fp']} {m_all['fn']} {m_all['tp']}")
