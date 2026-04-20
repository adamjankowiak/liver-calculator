import json
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, brier_score_loss, log_loss

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


POS_LABEL = "3-4"   # chory
NEG_LABEL = "1-2"   # zdrowy

SENS_TARGET = 0.95  # rule-out: sensitivity >= 0.95
SPEC_TARGET = 0.85  # rule-in:  specificity >= 0.85

CAL_METHOD = "sigmoid"   # "sigmoid" (Platt) albo "isotonic"
CAL_CV = 5               # folds w CalibratedClassifierCV
THR_OOF_FOLDS = 5        # OOF na TRAIN do wyznaczenia progów
RANDOM_STATE = 42

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

REPORT_PATH = "triage_fixed_full_report.txt"
MODEL_PATH  = "rf_calibrated_triage_model.joblib"
META_PATH   = "rf_calibrated_triage_meta.json"
RELIABILITY_PNG = "calibration_reliability.png"


#Przesiew i metryki
def confusion_from_threshold(y_true, p_pos, thr):
    y_pred = np.where(p_pos >= thr, POS_LABEL, NEG_LABEL)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL]).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def sens_spec_ppv_npv(tn, fp, fn, tp):
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) else 0.0
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    return float(sens), float(spec), float(ppv), float(npv)

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
            t_out = float(thr)

    for thr in thresholds:
        tn, fp, fn, tp = confusion_from_threshold(y_true, p_pos, thr)
        _, spec, *_ = sens_spec_ppv_npv(tn, fp, fn, tp)
        if spec >= spec_target:
            t_in = float(thr)
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
        tn_out = fn_out = 0
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
        tp_in = fp_in = 0
        ppv_in = 0.0
        fp_rate_in = 0.0

    # constraint check (binarne przy progach)
    tn_o, fp_o, fn_o, tp_o = confusion_from_threshold(y_true, p_pos, t_out)
    sens_o, spec_o, ppv_o, npv_o = sens_spec_ppv_npv(tn_o, fp_o, fn_o, tp_o)

    tn_i, fp_i, fn_i, tp_i = confusion_from_threshold(y_true, p_pos, t_in)
    sens_i, spec_i, ppv_i, npv_i = sens_spec_ppv_npv(tn_i, fp_i, fn_i, tp_i)

    return {
        "t_out": float(t_out),
        "t_in": float(t_in),
        "n": int(n),

        "out_frac": n_out / n if n else 0.0,
        "in_frac":  n_in / n if n else 0.0,
        "grey_frac": n_g / n if n else 0.0,

        "NPV_out": float(npv_out),
        "FN_rate_out": float(fn_rate_out),
        "PPV_in": float(ppv_in),
        "FP_rate_in": float(fp_rate_in),

        "at_t_out": {
            "tn": tn_o, "fp": fp_o, "fn": fn_o, "tp": tp_o,
            "sens": sens_o, "spec": spec_o, "ppv": ppv_o, "npv": npv_o
        },
        "at_t_in": {
            "tn": tn_i, "fp": fp_i, "fn": fn_i, "tp": tp_i,
            "sens": sens_i, "spec": spec_i, "ppv": ppv_i, "npv": npv_i
        },

        "subset_counts": {
            "OUT": {"n": n_out, "tn": tn_out, "fn": fn_out},
            "IN":  {"n": n_in,  "tp": tp_in, "fp": fp_in},
            "GREY": {"n": n_g},
        }
    }

def _pprint_triage(rep, title, fh):
    print(f"\n=== {title} ===", file=fh)
    print(f"t_out={rep['t_out']:.6f}, t_in={rep['t_in']:.6f}", file=fh)
    print(
        f"Coverage OUT/IN/GREY: {rep['out_frac']:.1%} / {rep['in_frac']:.1%} / {rep['grey_frac']:.1%}",
        file=fh
    )
    print(
        f"Rule-out safety (OUT): NPV_out={rep['NPV_out']:.4f}, FN_rate_out={rep['FN_rate_out']:.4f}",
        file=fh
    )
    print(
        f"Rule-in certainty (IN): PPV_in={rep['PPV_in']:.4f}, FP_rate_in={rep['FP_rate_in']:.4f}",
        file=fh
    )
    print(
        f"Constraint check: sens@t_out={rep['at_t_out']['sens']:.4f} (>= {SENS_TARGET}), "
        f"spec@t_in={rep['at_t_in']['spec']:.4f} (>= {SPEC_TARGET})",
        file=fh
    )

    print("\n-- Confusion/Metrics @ t_out (predict POS if p>=t_out) --", file=fh)
    a = rep["at_t_out"]
    print(f"TN={a['tn']}, FP={a['fp']}, FN={a['fn']}, TP={a['tp']}", file=fh)
    print(f"Sens={a['sens']:.4f}, Spec={a['spec']:.4f}, PPV={a['ppv']:.4f}, NPV={a['npv']:.4f}", file=fh)

    print("\n-- Confusion/Metrics @ t_in (predict POS if p>=t_in) --", file=fh)
    b = rep["at_t_in"]
    print(f"TN={b['tn']}, FP={b['fp']}, FN={b['fn']}, TP={b['tp']}", file=fh)
    print(f"Sens={b['sens']:.4f}, Spec={b['spec']:.4f}, PPV={b['ppv']:.4f}, NPV={b['npv']:.4f}", file=fh)

    print("\n-- Subset counts --", file=fh)
    sc = rep["subset_counts"]
    print(f"OUT: n={sc['OUT']['n']}, TN={sc['OUT'].get('tn',0)}, FN={sc['OUT'].get('fn',0)}", file=fh)
    print(f"IN:  n={sc['IN']['n']}, TP={sc['IN'].get('tp',0)}, FP={sc['IN'].get('fp',0)}", file=fh)
    print(f"GREY:n={sc['GREY']['n']}", file=fh)


#Kalibracja
def _to_binary(y, pos_label=POS_LABEL):
    y = np.asarray(y)
    return (y == pos_label).astype(int)

def calibration_table(y_true, p_pos, n_bins=10):

    yb = _to_binary(y_true)
    p = np.asarray(p_pos, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    total = len(p)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)

        n = int(np.sum(m))
        if n == 0:
            rows.append([lo, hi, 0, np.nan, np.nan, np.nan, 0.0])
            continue

        mean_pred = float(np.mean(p[m]))
        frac_pos = float(np.mean(yb[m]))
        gap = abs(mean_pred - frac_pos)
        contrib = (n / total) * gap
        ece += contrib
        rows.append([lo, hi, n, mean_pred, frac_pos, gap, contrib])

    df = pd.DataFrame(
        rows,
        columns=["bin_low", "bin_high", "n", "mean_pred", "frac_pos", "abs_gap", "ece_contrib"]
    )
    return df, float(ece)

def calibration_metrics(y_true, p_pos):
    yb = _to_binary(y_true)
    p = np.asarray(p_pos, dtype=float)

    # clipping dla logloss stabilności
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)

    brier = float(brier_score_loss(yb, p))
    ll = float(log_loss(yb, p_clip))
    tab, ece = calibration_table(y_true, p, n_bins=10)
    return {"brier": brier, "logloss": ll, "ece": ece, "table": tab}

def plot_reliability(dfs, labels, path_png=RELIABILITY_PNG):
    if not HAS_PLOT:
        return False

    plt.figure()
    # perfect line
    plt.plot([0, 1], [0, 1])

    for df, lab in zip(dfs, labels):
        m = df["n"] > 0
        x = df.loc[m, "mean_pred"].values
        y = df.loc[m, "frac_pos"].values
        plt.plot(x, y, marker="o", linestyle="-", label=lab)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability (Calibration) Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close()
    return True

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


#INPUT
train_df = pd.read_csv("Data/Modified/Full/Final_reduced/2-class/Scaled/patients.csv")
test_df  = pd.read_csv("Data/Modified/Full/Final_reduced/2-class/Scaled/control.csv")

target_col = "FSCORE"
feature_cols = [
    "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]",
    "FIB4", "APRI", "NAFLD", "RITIS"
]

train_df[target_col] = train_df[target_col].astype(str).str.strip()
test_df[target_col]  = test_df[target_col].astype(str).str.strip()

allowed = {NEG_LABEL, POS_LABEL}
train_df = train_df[train_df[target_col].isin(allowed)].copy()
test_df  = test_df[test_df[target_col].isin(allowed)].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test  = test_df[feature_cols].copy()
y_test  = test_df[target_col].copy()


p_train_oof = oof_calibrated_probs(X_train, y_train, RF_PARAMS, seed=RANDOM_STATE)
t_out, t_in = find_thresholds(y_train.values, p_train_oof)

if (t_out is None) or (t_in is None) or (t_out > t_in):
    raise RuntimeError("Nie da się wyznaczyć progów spełniających constrainty na TRAIN (OOF).")


rf_final = RandomForestClassifier(**RF_PARAMS)
cal_final = CalibratedClassifierCV(estimator=rf_final, method=CAL_METHOD, cv=CAL_CV)
cal_final.fit(X_train, y_train)

pos_idx = np.where(cal_final.classes_ == POS_LABEL)[0][0]

p_train_final = cal_final.predict_proba(X_train)[:, pos_idx]
p_test = cal_final.predict_proba(X_test)[:, pos_idx]


rep_train_oof = triage_report(y_train.values, p_train_oof, t_out, t_in)
rep_train_final = triage_report(y_train.values, p_train_final, t_out, t_in)
rep_test = triage_report(y_test.values, p_test, t_out, t_in)


cal_train_oof = calibration_metrics(y_train.values, p_train_oof)
cal_train_final = calibration_metrics(y_train.values, p_train_final)
cal_test = calibration_metrics(y_test.values, p_test)


rf_importances = None
try:
    rfs = [cc.estimator for cc in cal_final.calibrated_classifiers_]
    imps = np.vstack([rf.feature_importances_ for rf in rfs])
    rf_importances = imps.mean(axis=0)
except Exception:
    rf_importances = None


#Raport
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    print("FINAL TRIAGE + CALIBRATION REPORT (fixed params, no grid search)", file=f)
    print("=" * 78, file=f)

    print("\n[CONFIG]", file=f)
    print(f"POS_LABEL={POS_LABEL}, NEG_LABEL={NEG_LABEL}", file=f)
    print(f"Targets: Sens>= {SENS_TARGET}, Spec>= {SPEC_TARGET}", file=f)
    print(f"Calibration: method={CAL_METHOD}, CalCV={CAL_CV}", file=f)
    print(f"OOF folds for thresholds: {THR_OOF_FOLDS}", file=f)
    print(f"RANDOM_STATE={RANDOM_STATE}", file=f)

    print("\n[MODEL PARAMS]", file=f)
    print("RF_PARAMS:", RF_PARAMS, file=f)

    print("\n[DATA]", file=f)
    print(f"Train n={len(X_train)}, Test n={len(X_test)}", file=f)
    pos_rate_train = float(np.mean(_to_binary(y_train.values)))
    pos_rate_test = float(np.mean(_to_binary(y_test.values)))
    print(f"Prevalence (POS) train={pos_rate_train:.4f}, test={pos_rate_test:.4f}", file=f)
    print("Features:", feature_cols, file=f)

    print("\n[THRESHOLDS learned from TRAIN (OOF)]", file=f)
    print(f"t_out={t_out:.8f}", file=f)
    print(f"t_in ={t_in:.8f}", file=f)

    print("\n" + "-"*78, file=f)
    print("[TRIAGE RESULTS]", file=f)
    _pprint_triage(rep_train_oof, "TRAIN (OOF) TRIAGE REPORT", f)
    _pprint_triage(rep_train_final, "TRAIN (FINAL model, in-sample) TRIAGE REPORT (diagnostic)", f)
    _pprint_triage(rep_test, "TEST TRIAGE REPORT", f)

    print("\n" + "-"*78, file=f)
    print("[CALIBRATION RESULTS]", file=f)

    def _print_cal(name, cal):
        print(f"\n--- {name} ---", file=f)
        print(f"Brier={cal['brier']:.6f}", file=f)
        print(f"LogLoss={cal['logloss']:.6f}", file=f)
        print(f"ECE(10 bins)={cal['ece']:.6f}", file=f)
        print("\nReliability table (10 bins):", file=f)
        # format table
        df = cal["table"].copy()
        df["bin_low"] = df["bin_low"].map(lambda x: f"{x:.1f}")
        df["bin_high"] = df["bin_high"].map(lambda x: f"{x:.1f}")
        df["mean_pred"] = df["mean_pred"].map(lambda x: "-" if pd.isna(x) else f"{x:.4f}")
        df["frac_pos"] = df["frac_pos"].map(lambda x: "-" if pd.isna(x) else f"{x:.4f}")
        df["abs_gap"] = df["abs_gap"].map(lambda x: "-" if pd.isna(x) else f"{x:.4f}")
        df["ece_contrib"] = df["ece_contrib"].map(lambda x: f"{x:.6f}")
        print(df.to_string(index=False), file=f)

    _print_cal("TRAIN (OOF)", cal_train_oof)
    _print_cal("TRAIN (FINAL in-sample)", cal_train_final)
    _print_cal("TEST", cal_test)

    print("\n" + "-"*78, file=f)
    print("[FEATURE IMPORTANCE]", file=f)
    if rf_importances is None:
        print("Feature importance: unavailable (could not extract RF estimators).", file=f)
    else:
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": rf_importances})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        print(imp_df.to_string(index=False), file=f)

    print("\n" + "-"*78, file=f)
    if HAS_PLOT:
        ok = plot_reliability(
            dfs=[cal_train_oof["table"], cal_train_final["table"], cal_test["table"]],
            labels=["TRAIN OOF", "TRAIN final", "TEST"],
            path_png=RELIABILITY_PNG
        )
        if ok:
            print(f"[PLOT] Saved reliability plot to: {RELIABILITY_PNG}", file=f)
        else:
            print("[PLOT] Reliability plot not saved.", file=f)
    else:
        print("[PLOT] matplotlib not available -> no plot saved.", file=f)

print(f"Zapisano raport do pliku: {REPORT_PATH}")


joblib.dump(cal_final, MODEL_PATH)
print(f"Zapisano model do pliku: {MODEL_PATH}")

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
    "report_path": REPORT_PATH,
    "model_path": MODEL_PATH,
    "reliability_plot": RELIABILITY_PNG if HAS_PLOT else None,
    "calibration_summary": {
        "train_oof": {"brier": cal_train_oof["brier"], "logloss": cal_train_oof["logloss"], "ece": cal_train_oof["ece"]},
        "train_final": {"brier": cal_train_final["brier"], "logloss": cal_train_final["logloss"], "ece": cal_train_final["ece"]},
        "test": {"brier": cal_test["brier"], "logloss": cal_test["logloss"], "ece": cal_test["ece"]},
    },
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Zapisano meta do pliku: {META_PATH}")
