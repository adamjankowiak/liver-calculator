import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, brier_score_loss, log_loss

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


# =========================
# CONFIG (jak w meta-kalkulatorze)
# =========================
POS_LABEL = "3-4"   # chory
NEG_LABEL = "1-2"   # zdrowy

SENS_TARGET = 0.95  # rule-out: sensitivity >= 0.95
SPEC_TARGET = 0.85  # rule-in:  specificity >= 0.85

CAL_METHOD = "sigmoid"   # "sigmoid" (Platt) albo "isotonic"
CAL_CV = 5               # folds w CalibratedClassifierCV
THR_OOF_FOLDS = 5        # OOF na TRAIN do wyznaczenia progów
RANDOM_STATE = 42

# ŚCIEŻKI – ustaw pod siebie (jak w RF meta-kalkulatorze)
ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "Modified" / "Full" / "Final_reduced" / "2-class" / "Scaled" / "patients.csv"
TEST_PATH = ROOT_DIR / "data" / "Modified" / "Full" / "Final_reduced" / "2-class" / "Scaled" / "control.csv"

REPORT_PATH = ROOT_DIR / "reports" / "metrics" / "lr_calibrated_triage_report.txt"
MODEL_PATH = ROOT_DIR / "models" / "lr_calibrated_triage_model.joblib"
META_PATH = ROOT_DIR / "models" / "metadata" / "lr_calibrated_triage_meta.json"
RELIABILITY_PNG = ROOT_DIR / "reports" / "figures" / "lr_calibration_reliability.png"

target_col = "FSCORE"
feature_cols = [
    "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]",
    "FIB4", "APRI", "RITIS", "NAFLD"
]

poly_cols = ["FIB4", "APRI", "RITIS", "NAFLD"]  # wielomiany tylko na tych
# reszta (bez wielomianów)
other_cols = [c for c in feature_cols if c not in poly_cols]

# FINAL best params (od Ciebie)
LR_PARAMS = {
    "C": 10,
    "class_weight": "balanced",
    "penalty": "elasticnet",
    "l1_ratio": 0,
    "solver": "saga",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


for artifact_path in (REPORT_PATH, MODEL_PATH, META_PATH, RELIABILITY_PNG):
    artifact_path.parent.mkdir(parents=True, exist_ok=True)


def _repo_relative(path_obj):
    return path_obj.relative_to(ROOT_DIR).as_posix()


# =========================
# LABEL MAPPING (odporny jak w Twoim kodzie)
# =========================
def fscore_to_labels(v):
    if pd.isna(v):
        return np.nan
    try:
        fv = float(v)
        return POS_LABEL if fv >= 3 else NEG_LABEL
    except Exception:
        pass
    s = str(v).strip().lower()
    s = s.replace("–", "-").replace(" ", "")
    return POS_LABEL if s == "3-4" else (NEG_LABEL if s == "1-2" else np.nan)


# =========================
# TRIAGE / THRESHOLDS (jak w meta-kalkulatorze RF)
# =========================
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
    thresholds = np.unique(np.concatenate([np.asarray(p_pos, dtype=float), [0.0, 1.0]]))
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
    p_pos = np.asarray(p_pos, dtype=float)
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

    # constraint check przy progach
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


# =========================
# CALIBRATION METRICS (jak w meta-kalkulatorze RF)
# =========================
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
        m = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))

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
    plt.plot([0, 1], [0, 1])  # perfect line

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


# =========================
# MODEL: preprocess + poly + scaler + LR
# =========================
def build_base_estimator():
    preprocess = ColumnTransformer(
        transformers=[
            ("poly", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]), poly_cols),
            ("other", SimpleImputer(strategy="median"), other_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    base = Pipeline(steps=[
        ("preprocess", preprocess),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])
    return base


def _make_calibrator(estimator):
    # zgodność między wersjami sklearn (estimator vs base_estimator)
    try:
        return CalibratedClassifierCV(estimator=estimator, method=CAL_METHOD, cv=CAL_CV)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=CAL_METHOD, cv=CAL_CV)


def oof_calibrated_probs(X, y, base_estimator, seed=RANDOM_STATE):
    """OOF P(POS_LABEL) na TRAIN do wyznaczania progów bez leakage."""
    skf = StratifiedKFold(n_splits=THR_OOF_FOLDS, shuffle=True, random_state=seed)
    p_oof = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        est = clone(base_estimator)
        cal = _make_calibrator(est)
        cal.fit(X_tr, y_tr)

        pos_idx = np.where(cal.classes_ == POS_LABEL)[0][0]
        p_oof[va_idx] = cal.predict_proba(X_va)[:, pos_idx]

    return p_oof


def safe_load_dataset(path):
    df = pd.read_csv(path)
    y = df[target_col].apply(fscore_to_labels)
    m = ~y.isna()
    df = df.loc[m].copy()
    y = y.loc[m].astype(str)

    allowed = {NEG_LABEL, POS_LABEL}
    m2 = y.isin(allowed)
    df = df.loc[m2].copy()
    y = y.loc[m2].copy()

    X = df[feature_cols].copy()
    return X, y, df


# =========================
# RUN
# =========================
base_estimator = build_base_estimator()

X_train, y_train, train_df = safe_load_dataset(TRAIN_PATH)

if TEST_PATH:
    X_test, y_test, test_df = safe_load_dataset(TEST_PATH)
else:
    X_test, y_test, test_df = None, None, None

# 1) OOF kalibrowane prawdopodobieństwa na TRAIN -> progi
p_train_oof = oof_calibrated_probs(X_train, y_train, base_estimator, seed=RANDOM_STATE)
t_out, t_in = find_thresholds(y_train.values, p_train_oof)

if (t_out is None) or (t_in is None) or (t_out > t_in):
    raise RuntimeError("Nie da się wyznaczyć progów spełniających constrainty na TRAIN (OOF).")

# 2) FINAL kalibrowany model trenowany na CAŁYM TRAIN
cal_final = _make_calibrator(clone(base_estimator))
cal_final.fit(X_train, y_train)

pos_idx = np.where(cal_final.classes_ == POS_LABEL)[0][0]

p_train_final = cal_final.predict_proba(X_train)[:, pos_idx]
p_test = cal_final.predict_proba(X_test)[:, pos_idx] if X_test is not None else None

# 3) TRIAGE raporty
rep_train_oof = triage_report(y_train.values, p_train_oof, t_out, t_in)
rep_train_final = triage_report(y_train.values, p_train_final, t_out, t_in)
rep_test = triage_report(y_test.values, p_test, t_out, t_in) if p_test is not None else None

# 4) CALIBRATION metryki
cal_train_oof = calibration_metrics(y_train.values, p_train_oof)
cal_train_final = calibration_metrics(y_train.values, p_train_final)
cal_test = calibration_metrics(y_test.values, p_test) if p_test is not None else None

# 5) (opcjonalnie) współczynniki LR – diagnostycznie (średnia po foldach kalibracji)
coef_df = None
try:
    ccs = cal_final.calibrated_classifiers_
    rows = []
    feat_names = None

    for cc in ccs:
        est = getattr(cc, "estimator", None)
        if est is None:
            continue
        lr = est.named_steps["clf"]
        if feat_names is None:
            # nazwy cech po preprocess (scaler nie zmienia nazw)
            feat_names = est.named_steps["preprocess"].get_feature_names_out()
        coefs = lr.coef_.ravel()
        rows.append(coefs)

    if rows and (feat_names is not None):
        M = np.vstack(rows)
        coef_df = pd.DataFrame({
            "feature": feat_names,
            "coef_mean": M.mean(axis=0),
            "coef_std": M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1]),
        }).sort_values("coef_mean", ascending=False).reset_index(drop=True)
except Exception:
    coef_df = None

# 6) RAPORT do pliku
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    print("FINAL TRIAGE + CALIBRATION REPORT (LogReg + Poly + fixed params)", file=f)
    print("=" * 78, file=f)

    print("\n[CONFIG]", file=f)
    print(f"POS_LABEL={POS_LABEL}, NEG_LABEL={NEG_LABEL}", file=f)
    print(f"Targets: Sens>= {SENS_TARGET}, Spec>= {SPEC_TARGET}", file=f)
    print(f"Calibration: method={CAL_METHOD}, CalCV={CAL_CV}", file=f)
    print(f"OOF folds for thresholds: {THR_OOF_FOLDS}", file=f)
    print(f"RANDOM_STATE={RANDOM_STATE}", file=f)

    print("\n[MODEL PARAMS]", file=f)
    print("LR_PARAMS:", LR_PARAMS, file=f)
    print("poly_cols:", poly_cols, file=f)

    print("\n[DATA]", file=f)
    print(f"Train n={len(X_train)}", file=f)
    if X_test is not None:
        print(f"Test  n={len(X_test)}", file=f)

    pos_rate_train = float(np.mean(_to_binary(y_train.values)))
    print(f"Prevalence (POS) train={pos_rate_train:.4f}", file=f)
    if y_test is not None:
        pos_rate_test = float(np.mean(_to_binary(y_test.values)))
        print(f"Prevalence (POS) test ={pos_rate_test:.4f}", file=f)

    print("Features:", feature_cols, file=f)

    print("\n[THRESHOLDS learned from TRAIN (OOF)]", file=f)
    print(f"t_out={t_out:.8f}", file=f)
    print(f"t_in ={t_in:.8f}", file=f)

    print("\n" + "-"*78, file=f)
    print("[TRIAGE RESULTS]", file=f)
    _pprint_triage(rep_train_oof, "TRAIN (OOF) TRIAGE REPORT", f)
    _pprint_triage(rep_train_final, "TRAIN (FINAL model, in-sample) TRIAGE REPORT (diagnostic)", f)
    if rep_test is not None:
        _pprint_triage(rep_test, "TEST TRIAGE REPORT", f)

    print("\n" + "-"*78, file=f)
    print("[CALIBRATION RESULTS]", file=f)

    def _print_cal(name, cal):
        print(f"\n--- {name} ---", file=f)
        print(f"Brier={cal['brier']:.6f}", file=f)
        print(f"LogLoss={cal['logloss']:.6f}", file=f)
        print(f"ECE(10 bins)={cal['ece']:.6f}", file=f)
        print("\nReliability table (10 bins):", file=f)
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
    if cal_test is not None:
        _print_cal("TEST", cal_test)

    print("\n" + "-"*78, file=f)
    print("[LR COEFFICIENTS (diagnostic)]", file=f)
    if coef_df is None:
        print("Coefficients: unavailable (could not extract from calibrated folds).", file=f)
    else:
        print(coef_df.to_string(index=False), file=f)

    print("\n" + "-"*78, file=f)
    if HAS_PLOT:
        ok = plot_reliability(
            dfs=[cal_train_oof["table"], cal_train_final["table"]] + ([cal_test["table"]] if cal_test is not None else []),
            labels=["TRAIN OOF", "TRAIN final"] + (["TEST"] if cal_test is not None else []),
            path_png=RELIABILITY_PNG
        )
        if ok:
            print(f"[PLOT] Saved reliability plot to: {RELIABILITY_PNG}", file=f)
        else:
            print("[PLOT] Reliability plot not saved.", file=f)
    else:
        print("[PLOT] matplotlib not available -> no plot saved.", file=f)

print(f"Zapisano raport do pliku: {REPORT_PATH}")

# 7) zapis modelu + meta
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
    "RANDOM_STATE": RANDOM_STATE,

    "LR_PARAMS": LR_PARAMS,
    "feature_cols": feature_cols,
    "poly_cols": poly_cols,

    "t_out": float(t_out),
    "t_in": float(t_in),

    "train_path": _repo_relative(TRAIN_PATH),
    "test_path": _repo_relative(TEST_PATH) if TEST_PATH else None,

    "report_path": _repo_relative(REPORT_PATH),
    "model_path": _repo_relative(MODEL_PATH),
    "reliability_plot": (_repo_relative(RELIABILITY_PNG) if HAS_PLOT else None),

    "calibration_summary": {
        "train_oof": {"brier": cal_train_oof["brier"], "logloss": cal_train_oof["logloss"], "ece": cal_train_oof["ece"]},
        "train_final": {"brier": cal_train_final["brier"], "logloss": cal_train_final["logloss"], "ece": cal_train_final["ece"]},
        "test": ({"brier": cal_test["brier"], "logloss": cal_test["logloss"], "ece": cal_test["ece"]} if cal_test is not None else None),
    },
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Zapisano meta do pliku: {META_PATH}")

# =========================
# (opcjonalnie) mini-demo inference na 1 pacjencie:
# - w produkcji: ładujesz MODEL_PATH i META_PATH, tworzysz DataFrame z feature_cols
# =========================
# cal = joblib.load(MODEL_PATH)
# with open(META_PATH, "r", encoding="utf-8") as f:
#     m = json.load(f)
# x_one = pd.DataFrame([{c: 0.0 for c in m["feature_cols"]}])  # <- tu podstaw realne wartości
# pos_idx = list(cal.classes_).index(m["POS_LABEL"])
# p = cal.predict_proba(x_one)[:, pos_idx][0]
# tri = "OUT" if p < m["t_out"] else ("IN" if p >= m["t_in"] else "GREY")
# print("p_pos=", p, "triage=", tri)
