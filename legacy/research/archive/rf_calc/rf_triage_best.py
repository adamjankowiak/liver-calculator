# rf_triage_best_only_fast_ece_brier.py
# RF + calibration
# Reports only: ECE + Brier for TRAIN (OOF), TRAIN (in-sample), TEST

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

# =========================
#  KONFIGURACJA
# =========================
POS_LABEL = "3-4"
NEG_LABEL = "1-2"

CAL_METHOD = "sigmoid"   # "sigmoid" albo "isotonic"
CAL_CV = 5
THR_OOF_FOLDS = 5
RANDOM_STATE = 42

BEST_PARAMS = {
    "bootstrap": True,
    "class_weight": "balanced",
    "max_depth": 4,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "min_samples_split": 10,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": 42,
}

TRAIN_PATH = "Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv"
TEST_PATH  = "Data/Test-group/control-imp-2-class.csv"

REPORT_PATH = "ece_brier_report.txt"

target_col = "FSCORE"
feature_cols = ["Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "NAFLD", "RITIS"]

# =========================
#  METRYKI KALIBRACJI
# =========================
def calibration_metrics(y_true, p_pos, n_bins=10):
    """
    Zwraca (ECE, Brier Score) dla klasy pozytywnej.
    ECE = sum_k (n_k/n) * |acc_k - conf_k|
    """
    y_true = np.asarray(y_true)
    y_bin = (y_true == POS_LABEL).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)

    # Brier score (im mniejszy tym lepiej)
    brier = brier_score_loss(y_bin, p_pos)

    # ECE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_pos)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # ostatni bin domykamy z prawej
        if i == n_bins - 1:
            mask = (p_pos >= lo) & (p_pos <= hi)
        else:
            mask = (p_pos >= lo) & (p_pos < hi)

        if not np.any(mask):
            continue

        p_bin = p_pos[mask]
        y_bin_i = y_bin[mask]

        conf = float(np.mean(p_bin))     # średnie przewidywane p w binie
        acc  = float(np.mean(y_bin_i))   # częstość pozytywów w binie
        ece += (len(p_bin) / n) * abs(acc - conf)

    return float(ece), float(brier)

# =========================
#  OOF calibrated probs (bez leakage)
# =========================
def oof_calibrated_probs(X, y, rf_params, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=THR_OOF_FOLDS, shuffle=True, random_state=seed)
    p_oof = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        rf = RandomForestClassifier(**rf_params)
        cal = CalibratedClassifierCV(estimator=rf, method=CAL_METHOD, cv=CAL_CV)
        cal.fit(X_tr, y_tr)

        pos_idx = np.where(cal.classes_ == POS_LABEL)[0][0]
        p_oof[va_idx] = cal.predict_proba(X_va)[:, pos_idx]

    return p_oof

# =========================
#  MAIN
# =========================
def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    train_df[target_col] = train_df[target_col].astype(str).str.strip()
    test_df[target_col]  = test_df[target_col].astype(str).str.strip()

    allowed = {NEG_LABEL, POS_LABEL}
    train_df = train_df[train_df[target_col].isin(allowed)].copy()
    test_df  = test_df[test_df[target_col].isin(allowed)].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[target_col].copy()

    # TRAIN OOF (uczciwe)
    p_train_oof = oof_calibrated_probs(X_train, y_train, BEST_PARAMS, seed=RANDOM_STATE)

    # Final model na całym TRAIN (in-sample + TEST)
    rf_final = RandomForestClassifier(**BEST_PARAMS)
    cal_final = CalibratedClassifierCV(estimator=rf_final, method=CAL_METHOD, cv=CAL_CV)
    cal_final.fit(X_train, y_train)
    pos_idx = np.where(cal_final.classes_ == POS_LABEL)[0][0]

    p_train_in = cal_final.predict_proba(X_train)[:, pos_idx]
    p_test     = cal_final.predict_proba(X_test)[:, pos_idx]

    # Metryki
    ece_train_oof, brier_train_oof = calibration_metrics(y_train.values, p_train_oof, n_bins=10)
    ece_train_in,  brier_train_in  = calibration_metrics(y_train.values, p_train_in,  n_bins=10)
    ece_test,      brier_test      = calibration_metrics(y_test.values,  p_test,      n_bins=10)

    # Raport
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        print("CALIBRATION REPORT (ECE + BRIER)", file=f)
        print("=" * 80, file=f)
        print("Best params:", BEST_PARAMS, file=f)
        print(f"Calibration method: {CAL_METHOD}, CalCV: {CAL_CV}", file=f)
        print(f"ECE bins: 10", file=f)

        print("\n--- TRAIN (OOF, no leakage) ---", file=f)
        print(f"ECE:   {ece_train_oof:.6f}", file=f)
        print(f"Brier: {brier_train_oof:.6f}", file=f)

        print("\n--- TRAIN (in-sample, diagnostic) ---", file=f)
        print(f"ECE:   {ece_train_in:.6f}", file=f)
        print(f"Brier: {brier_train_in:.6f}", file=f)

        print("\n--- TEST ---", file=f)
        print(f"ECE:   {ece_test:.6f}", file=f)
        print(f"Brier: {brier_test:.6f}", file=f)

    print(f"Zapisano raport: {REPORT_PATH}")

if __name__ == "__main__":
    main()
