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

RF_PARAMS = {
    "bootstrap": True,
    "class_weight": "balanced",
    "max_depth": 6,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

# RF_PARAMS = {
#     "bootstrap": True,
#     "class_weight": "balanced",
#     "max_depth": 4,
#     "max_features": "sqrt",
#     "min_samples_leaf": 2,
#     "min_samples_split": 10,
#     "n_estimators": 100,
#     "random_state": 42,
#     "n_jobs": -1,
# }

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

def find_thresholds(y_true, p_pos, sens_target=0.95, spec_target=0.85):
    """
    t_out: największy próg z sensitivity >= sens_target
    t_in:  najmniejszy próg z specificity >= spec_target
    """
    thresholds = np.unique(np.concatenate([p_pos, [0.0, 1.0]]))
    thresholds.sort()

    t_out = None
    t_in = None

    # rule-out: największy próg z sens >= target
    for thr in thresholds:
        tn, fp, fn, tp = confusion_from_threshold(y_true, p_pos, thr)
        sens, spec, ppv, npv = sens_spec_ppv_npv(tn, fp, fn, tp)
        if sens >= sens_target:
            t_out = thr

    # rule-in: najmniejszy próg z spec >= target
    for thr in thresholds:
        tn, fp, fn, tp = confusion_from_threshold(y_true, p_pos, thr)
        sens, spec, ppv, npv = sens_spec_ppv_npv(tn, fp, fn, tp)
        if spec >= spec_target:
            t_in = thr
            break

    return t_out, t_in

def triage_assign(p_pos, t_out, t_in):
    triage = np.full(len(p_pos), "GREY", dtype=object)
    triage[p_pos < t_out] = "OUT"   # rule-out (zdrowy)
    triage[p_pos >= t_in] = "IN"    # rule-in  (chory)
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

    # klasyczne metryki dla progów jako binarne
    tn_o, fp_o, fn_o, tp_o = confusion_from_threshold(y_true, p_pos, t_out)
    sens_o, spec_o, ppv_o, npv_o = sens_spec_ppv_npv(tn_o, fp_o, fn_o, tp_o)

    tn_i, fp_i, fn_i, tp_i = confusion_from_threshold(y_true, p_pos, t_in)
    sens_i, spec_i, ppv_i, npv_i = sens_spec_ppv_npv(tn_i, fp_i, fn_i, tp_i)

    return {
        "t_out": float(t_out),
        "t_in": float(t_in),

        "n": int(n),
        "out_count": n_out,
        "in_count": n_in,
        "grey_count": n_g,
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

def triage_score_min_grey(report, sens_target=0.95, spec_target=0.85):
    ok = (report["sens_at_t_out"] >= sens_target) and (report["spec_at_t_in"] >= spec_target)
    if not ok:
        sens_gap = max(0.0, sens_target - report["sens_at_t_out"])
        spec_gap = max(0.0, spec_target - report["spec_at_t_in"])
        return 1.0 + sens_gap + spec_gap
    return report["grey_frac"]

def triage_score_utility(report, sens_target=0.95, spec_target=0.85,
                        cost_fn_out=20.0, cost_fp_in=5.0, cost_grey=1.0):
    cost = (
        cost_fn_out * report["FN_rate_out"] * report["out_frac"] +
        cost_fp_in  * report["FP_rate_in"] * report["in_frac"] +
        cost_grey   * report["grey_frac"]
    )
    sens_gap = max(0.0, sens_target - report["sens_at_t_out"])
    spec_gap = max(0.0, spec_target - report["spec_at_t_in"])
    penalty = 50.0 * (sens_gap + spec_gap)
    return cost + penalty


# =========================
#  DANE
# =========================
train_df = pd.read_csv("Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv")
test_df  = pd.read_csv("Data/Test-group/control-imp-2-class.csv")

target_col = "FSCORE"
#feature_cols = ["FIB4", "RITIS", "APRI", "NAFLD"]
feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "NAFLD", "RITIS"]

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
#  KROK 1: OOF skalibrowane prawdopodobieństwa na TRAIN (do progów)
# =========================
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cal = 5

p_oof = np.zeros(len(y_train), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(outer.split(X_train, y_train), start=1):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]

    rf = RandomForestClassifier(**RF_PARAMS)
    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=inner_cal)
    cal.fit(X_tr, y_tr)

    pos_idx = np.where(cal.classes_ == POS_LABEL)[0][0]
    p_oof[va_idx] = cal.predict_proba(X_va)[:, pos_idx]


# =========================
#  KROK 2: Progi pod triage (na OOF)
# =========================
t_out, t_in = find_thresholds(y_train.values, p_oof, sens_target=SENS_TARGET, spec_target=SPEC_TARGET)

if t_out is None or t_in is None:
    raise RuntimeError("Nie da się wyznaczyć progów spełniających constrainty na train (OOF).")

if t_out > t_in:
    raise RuntimeError(
        f"Sprzeczne progi: t_out={t_out:.4f} > t_in={t_in:.4f}. "
        "Model nie daje sensownej strefy szarej."
    )

train_rep = triage_report(y_train.values, p_oof, t_out, t_in)

print("\n=== TRAIN (OOF) TRIAGE REPORT ===")
print(f"t_out={train_rep['t_out']:.4f}, t_in={train_rep['t_in']:.4f}")
print(f"Coverage OUT/IN/GREY: {train_rep['out_frac']:.1%} / {train_rep['in_frac']:.1%} / {train_rep['grey_frac']:.1%}")
print(f"Rule-out safety: NPV_out={train_rep['NPV_out']:.4f}, FN_rate_out={train_rep['FN_rate_out']:.4f}")
print(f"Rule-in certainty: PPV_in={train_rep['PPV_in']:.4f}, FP_rate_in={train_rep['FP_rate_in']:.4f}")
print(f"Constraints check: sens@t_out={train_rep['sens_at_t_out']:.4f} (>= {SENS_TARGET}), "
      f"spec@t_in={train_rep['spec_at_t_in']:.4f} (>= {SPEC_TARGET})")

print("Triage score (min grey):", f"{triage_score_min_grey(train_rep, SENS_TARGET, SPEC_TARGET):.4f}")
print("Triage score (utility): ", f"{triage_score_utility(train_rep, SENS_TARGET, SPEC_TARGET):.4f}")


# =========================
#  KROK 3: Finalny model + kalibrator na CAŁYM TRAIN
# =========================
rf_final = RandomForestClassifier(**RF_PARAMS)
cal_final = CalibratedClassifierCV(estimator=rf_final, method="sigmoid", cv=5)
cal_final.fit(X_train, y_train)

# =========================
#  KROK 4: TEST – triage raport
# =========================
pos_idx = np.where(cal_final.classes_ == POS_LABEL)[0][0]
p_test = cal_final.predict_proba(X_test)[:, pos_idx]

test_rep = triage_report(y_test.values, p_test, t_out, t_in)

print("\n=== TEST TRIAGE REPORT ===")
print(f"t_out={test_rep['t_out']:.4f}, t_in={test_rep['t_in']:.4f}")
print(f"Coverage OUT/IN/GREY: {test_rep['out_frac']:.1%} / {test_rep['in_frac']:.1%} / {test_rep['grey_frac']:.1%}")
print(f"Rule-out safety: NPV_out={test_rep['NPV_out']:.4f}, FN_rate_out={test_rep['FN_rate_out']:.4f}")
print(f"Rule-in certainty: PPV_in={test_rep['PPV_in']:.4f}, FP_rate_in={test_rep['FP_rate_in']:.4f}")
print(f"Constraints check: sens@t_out={test_rep['sens_at_t_out']:.4f} (>= {SENS_TARGET}), "
      f"spec@t_in={test_rep['spec_at_t_in']:.4f} (>= {SPEC_TARGET})")

print("Triage score (min grey):", f"{triage_score_min_grey(test_rep, SENS_TARGET, SPEC_TARGET):.4f}")
print("Triage score (utility): ", f"{triage_score_utility(test_rep, SENS_TARGET, SPEC_TARGET):.4f}")
