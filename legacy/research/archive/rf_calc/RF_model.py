# random_forest_triage_fixed.py
# Fixed RandomForest params + calibration + dual-threshold triage (rule-out/grey/rule-in)
# No GridSearch: uses RF_PARAMS exactly as provided.

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


# =========================
#  KONFIGURACJA
# =========================
POS_LABEL = "3-4"   # chory
NEG_LABEL = "1-2"   # zdrowy

SENS_TARGET = 0.95  # rule-out: sensitivity >= 0.95
SPEC_TARGET = 0.85  # rule-in:  specificity >= 0.85

CAL_METHOD = "sigmoid"   # "sigmoid" (Platt) albo "isotonic"
CAL_CV = 5               # folds w CalibratedClassifierCV
THR_OOF_FOLDS = 5        # OOF na TRAIN do wyznaczenia progów
RANDOM_STATE = 42

# Twoje stałe parametry lasu:
RF_PARAMS = {
    "bootstrap": True,
    "class_weight": "balanced",
    "max_depth": 4,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "min_samples_split": 10,
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

# Ścieżki zapisu
REPORT_PATH = "triage_fixed_report.txt"
MODEL_PATH  = "rf_calibrated_triage_model.joblib"
META_PATH   = "rf_calibrated_triage_meta.json"


# =========================
#  FUNKCJE
# =========================
def confusion_from_threshold(y_true, p_pos, thr):
    y_pred = np.where(p_pos >= thr, POS_LABEL, NEG_LABEL)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL]).ravel()
    return tn, fp, fn, tp

def sens_spec_ppv_npv(tn, fp, fn, tp):
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    return sens, spec, ppv, npv

def find_thresholds(y_true, p_pos, sens_target=SENS_TARGET, spec_target=SPEC_TARGET):
    """
    t_out: największy próg z sensitivity >= sens_target
    t_in:  najmniejszy próg z specificity >= spec_target
    """
    thresholds = np.unique(np.concatenate([p_pos, [0.0, 1.0]]))
    thresholds.sort()

    t_out = None
    t_in = None

    for thr in thresholds:
        tn, fp, fn, tp = confusion_from_threshold(y_true, p_pos, thr)
        sens, spec, *_ = sens_spec_ppv_npv(tn, fp, fn, tp)
        if sens >= sens_target:
            t_out = thr

    for thr in thresholds:
        tn, fp, fn, tp = confusion_from_threshold(y_true, p_pos, thr)
        _, spec, *_ = sens_spec_ppv_npv(tn, fp, fn, tp)
        if spec >= spec_target:
            t_in = thr
            break

    return t_out, t_in

def triage_assign(p_pos, t_out, t_in):
    triage = np.full(len(p_pos), "GREY", dtype=object)
    triage[p_pos < t_out] = "OUT"
    triage[p_pos >= t_in] = "IN"
    return triage

def triage_report(y_true, p_pos, t_out, t_in):
    y_true = np.asarray(y_true)
    tri = triage_assign(p_pos, t_out, t_in)

    n = len(y_true)
    n_out = int(np.sum(tri == "OUT"))
    n_in  = int(np.sum(tri == "IN"))
    n_g   = int(np.sum(tri == "GREY"))

    # OUT subset (pred=NEG)
    if n_out > 0:
        y_out = y_true[tri == "OUT"]
        tn_out = int(np.sum(y_out == NEG_LABEL))
        fn_out = int(np.sum(y_out == POS_LABEL))
        npv_out = tn_out / (tn_out + fn_out) if (tn_out + fn_out) else 0.0
        fn_rate_out = fn_out / n_out
    else:
        npv_out = 0.0
        fn_rate_out = 0.0

    # IN subset (pred=POS)
    if n_in > 0:
        y_in = y_true[tri == "IN"]
        tp_in = int(np.sum(y_in == POS_LABEL))
        fp_in = int(np.sum(y_in == NEG_LABEL))
        ppv_in = tp_in / (tp_in + fp_in) if (tp_in + fp_in) else 0.0
        fp_rate_in = fp_in / n_in
    else:
        ppv_in = 0.0
        fp_rate_in = 0.0

    # constraint check (binarne przy progach)
    tn_o, fp_o, fn_o, tp_o = confusion_from_threshold(y_true, p_pos, t_out)
    sens_o, *_ = sens_spec_ppv_npv(tn_o, fp_o, fn_o, tp_o)

    tn_i, fp_i, fn_i, tp_i = confusion_from_threshold(y_true, p_pos, t_in)
    _, spec_i, *_ = sens_spec_ppv_npv(tn_i, fp_i, fn_i, tp_i)

    return {
        "t_out": float(t_out),
        "t_in": float(t_in),
        "n": int(n),
        "out_frac": n_out / n,
        "in_frac": n_in / n,
        "grey_frac": n_g / n,
        "NPV_out": float(npv_out),
        "FN_rate_out": float(fn_rate_out),
        "PPV_in": float(ppv_in),
        "FP_rate_in": float(fp_rate_in),
        "sens_at_t_out": float(sens_o),
        "spec_at_t_in": float(spec_i),
    }

def print_triage(rep, title, fh=None):
    out = fh if fh is not None else None
    def _p(s=""):
        print(s, file=out) if out is not None else print(s)

    _p(f"\n=== {title} ===")
    _p(f"t_out={rep['t_out']:.4f}, t_in={rep['t_in']:.4f}")
    _p(f"Coverage OUT/IN/GREY: {rep['out_frac']:.1%} / {rep['in_frac']:.1%} / {rep['grey_frac']:.1%}")
    _p(f"Rule-out safety: NPV_out={rep['NPV_out']:.4f}, FN_rate_out={rep['FN_rate_out']:.4f}")
    _p(f"Rule-in certainty: PPV_in={rep['PPV_in']:.4f}, FP_rate_in={rep['FP_rate_in']:.4f}")
    _p(f"Constraints check: sens@t_out={rep['sens_at_t_out']:.4f} (>= {SENS_TARGET}), "
       f"spec@t_in={rep['spec_at_t_in']:.4f} (>= {SPEC_TARGET})")

def oof_calibrated_probs(X, y, rf_params, seed=RANDOM_STATE):
    """OOF P(POS_LABEL) na TRAIN do wyznaczania progów bez leakage."""
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
#  DANE
# =========================
train_df = pd.read_csv("Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv")
test_df  = pd.read_csv("Data/Test-group/control-imp-2-class.csv")

target_col = "FSCORE"
# feature_cols = ["FIB4", "RITIS", "APRI", "NAFLD"]
feature_cols = ["Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]",
                "FIB4", "APRI", "NAFLD", "RITIS"]

train_df[target_col] = train_df[target_col].astype(str).str.strip()
test_df[target_col]  = test_df[target_col].astype(str).str.strip()

allowed = {NEG_LABEL, POS_LABEL}
train_df = train_df[train_df[target_col].isin(allowed)].copy()
test_df  = test_df[test_df[target_col].isin(allowed)].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test  = test_df[feature_cols].copy()
y_test  = test_df[target_col].copy()


# =========================
#  KROK 1: progi z TRAIN (OOF)
# =========================
p_train_oof = oof_calibrated_probs(X_train, y_train, RF_PARAMS, seed=RANDOM_STATE)
t_out, t_in = find_thresholds(y_train.values, p_train_oof)

if (t_out is None) or (t_in is None) or (t_out > t_in):
    raise RuntimeError("Nie da się wyznaczyć progów spełniających constrainty na TRAIN (OOF).")


# =========================
#  KROK 2: fit final model na całym TRAIN
# =========================
rf_final = RandomForestClassifier(**RF_PARAMS)
cal_final = CalibratedClassifierCV(estimator=rf_final, method=CAL_METHOD, cv=CAL_CV)
cal_final.fit(X_train, y_train)

pos_idx = np.where(cal_final.classes_ == POS_LABEL)[0][0]


# =========================
#  RAPORTY + ZAPIS
# =========================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    print("FINAL TRIAGE REPORT (fixed params, no grid search)", file=f)
    print("=" * 60, file=f)
    print("RF_PARAMS:", RF_PARAMS, file=f)
    print(f"Calibration: method={CAL_METHOD}, CalCV={CAL_CV}", file=f)
    print(f"Targets: Sens>= {SENS_TARGET}, Spec>= {SPEC_TARGET}", file=f)
    print("\nThresholds learned from TRAIN (OOF):", file=f)
    print(f"t_out={t_out}, t_in={t_in}", file=f)

    # TRAIN (OOF) — uczciwy na train
    rep_train_oof = triage_report(y_train.values, p_train_oof, t_out, t_in)
    print_triage(rep_train_oof, "TRAIN (OOF) TRIAGE REPORT", fh=f)

    # TRAIN (FINAL in-sample) — diagnostyczny
    p_train_final = cal_final.predict_proba(X_train)[:, pos_idx]
    rep_train_final = triage_report(y_train.values, p_train_final, t_out, t_in)
    print_triage(rep_train_final, "TRAIN (FINAL model, in-sample) TRIAGE REPORT (diagnostic)", fh=f)

    # TEST
    p_test = cal_final.predict_proba(X_test)[:, pos_idx]
    rep_test = triage_report(y_test.values, p_test, t_out, t_in)
    print_triage(rep_test, "TEST TRIAGE REPORT", fh=f)

print(f"Zapisano raport do pliku: {REPORT_PATH}")

# Zapis modelu (skalibrowany klasyfikator)
joblib.dump(cal_final, MODEL_PATH)
print(f"Zapisano model do pliku: {MODEL_PATH}")

# Zapis metadanych (progi + feature list)
meta = {
    "POS_LABEL": POS_LABEL,
    "NEG_LABEL": NEG_LABEL,
    "SENS_TARGET": SENS_TARGET,
    "SPEC_TARGET": SPEC_TARGET,
    "CAL_METHOD": CAL_METHOD,
    "CAL_CV": CAL_CV,
    "THR_OOF_FOLDS": THR_OOF_FOLDS,
    "RF_PARAMS": RF_PARAMS,
    "feature_cols": feature_cols,
    "t_out": float(t_out),
    "t_in": float(t_in),
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Zapisano meta do pliku: {META_PATH}")
