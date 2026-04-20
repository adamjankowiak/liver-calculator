"""
Hepatology meta-calculator — FSCORE 1 vs 2–4 (≥F2), DUAL THRESHOLDS
• Ensemble (GB + RF [+ XGB]), META-calibration (Isotonic) on ensemble output
• Two thresholds:
    - T_low  (rule-out): recall ≥ target_recall_low  AND NPV ≥ target_npv_low
                         + minimal rule-out fraction; fallback: Fβ (β=2)
    - T_high (rule-in):  specificity ≥ target_specificity_high + minimal rule-in fraction
• Consistent metrics & plots: CMs for T_low and T_high, ROC, histogram with both lines
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    average_precision_score, brier_score_loss
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ================== Helpers ==================

def _ece_score(y_true, y_proba, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        conf = y_proba[m].mean()
        acc = y_true[m].mean()
        ece += (np.sum(m) / len(y_true)) * abs(acc - conf)
    return float(ece)

def _metrics_from_counts(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    ppv  = prec
    npv  = tn / (tn + fn) if (tn + fn) else 0.0
    return prec, rec, spec, f1, ppv, npv

def _fbeta(prec, rec, beta=2.0):
    if prec == 0.0 or rec == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


# ================== Model ==================

class LiverFibrosisMetaCalculatorF2plusDual:
    """
    Classification:  FSCORE = 1  vs  FSCORE ∈ {2,3,4}
    (i.e., none/mild fibrosis vs significant fibrosis ≥F2).
    Mode: TWO THRESHOLDS (rule-out + rule-in) + isotonic META-calibration.
    """

    def __init__(
        self,
        target_recall_low=0.95,             # rule-out target (sensitivity)
        target_npv_low=0.95,                # rule-out target (NPV)
        target_specificity_high=0.80,       # rule-in target (specificity)
        beta_low=2.0,                       # fallback F-beta for T_low
        weight_grid=(5, 10, 15),            # class-1 (≥F2) weights
        n_splits_oof=5,
        random_state=42,
        min_ruleout_frac=0.10,              # minimum rule-out fraction (e.g., ≥10%)
        min_rulein_frac=0.05,               # minimum rule-in fraction
        clip_eps=1e-6,
        ensemble_weights=(0.5, 0.35, 0.15), # GB, RF, XGB (if no XGB → weights renormalize)
        meta_calibration="isotonic"         # 'isotonic' or 'none'
    ):
        self.target_recall_low = target_recall_low
        self.target_npv_low = target_npv_low
        self.target_specificity_high = target_specificity_high
        self.beta_low = beta_low
        self.weight_grid = list(weight_grid)
        self.n_splits_oof = n_splits_oof
        self.random_state = random_state
        self.min_ruleout_frac = min_ruleout_frac
        self.min_rulein_frac = min_rulein_frac
        self.clip_eps = clip_eps
        self.ensemble_weights = ensemble_weights
        self.meta_calibration = meta_calibration

        # Base estimators (no calibration!):
        self.gb_params = dict(
            n_estimators=300, learning_rate=0.02,
            max_depth=4, min_samples_split=10, min_samples_leaf=5,
            subsample=0.8, max_features='sqrt',
            random_state=self.random_state, validation_fraction=0.2,
            n_iter_no_change=20, tol=1e-4
        )
        self.rf_params = dict(
            n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2,
            max_features='sqrt', n_jobs=-1, random_state=self.random_state, bootstrap=True
        )
        self.xgb_params = dict(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=self.random_state, objective='binary:logistic', eval_metric='logloss'
        )

        # Post-training state
        self.best_weight = None
        self.sample_weights = None

        self.imputer = SimpleImputer(strategy='median')   # fitted on full-train
        self.models_ = []                                 # base-model list
        self.model_weights_ = []                          # their weights
        self.meta_calibrator_ = None                      # IsotonicRegression

        self.threshold_low = None
        self.threshold_high = None
        self.cutoff_low = None
        self.cutoff_high = None

        self.key_features = [
            'max_score','APRI','FIB4','weighted_sum','apri_x_ritis','RITIS',
            'apri_x_fib4','min_score','std_score','fib4_x_nafld'
        ]

    # ---------- Data / features ----------

    def prepare_data(self, df: pd.DataFrame):
        req = ['APRI', 'FIB4', 'RITIS', 'NAFLD', 'FSCORE']
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in the data")

        fs = pd.to_numeric(df['FSCORE'], errors='coerce')
        bad = fs.isna().sum()
        if bad:
            raise ValueError(f"FSCORE contains {bad} non-parsable values (expected 1/2/3/4).")
        unique_vals = np.sort(fs.unique())

        # target: 0 for 1, 1 for 2–4
        y = np.where(fs == 1, 0, 1).astype(int)
        X = df[['APRI', 'FIB4', 'RITIS', 'NAFLD']].copy()

        print(f"   Unique FSCORE values: {unique_vals}")
        print("   Mapping: 1 → 0 (FS1), 2–4 → 1 (≥F2)")
        return X, y

    def _safe_div(self, a, b):
        b2 = np.where(b == 0, np.nan, b)
        return a / b2

    def create_advanced_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_feat = X.copy()
        # statistics
        X_feat['mean_score'] = X.mean(axis=1)
        X_feat['max_score']  = X.max(axis=1)
        X_feat['min_score']  = X.min(axis=1)
        X_feat['std_score']  = X.std(axis=1)
        # normalized scales
        X_norm = X.copy()
        X_norm['APRI']  = X['APRI'] / 2.0
        X_norm['FIB4']  = X['FIB4'] / 10.0
        X_norm['RITIS'] = X['RITIS'] / 3.0
        X_norm['NAFLD'] = (X['NAFLD'] + 5) / 10.0
        # alarms
        X_feat['apri_alarm']  = (X['APRI']  >= 0.7).astype(int)
        X_feat['fib4_alarm']  = (X['FIB4']  >= 2.67).astype(int)
        X_feat['ritis_alarm'] = (X['RITIS'] >= 1.5).astype(int)
        X_feat['nafld_alarm'] = (X['NAFLD'] >= 0.676).astype(int)
        X_feat['n_alarms'] = X_feat[['apri_alarm','fib4_alarm','ritis_alarm','nafld_alarm']].sum(axis=1)
        X_feat['unanimous_alarm'] = (X_feat['n_alarms'] == 4).astype(int)
        X_feat['majority_alarm']  = (X_feat['n_alarms'] >= 3).astype(int)
        X_feat['any_alarm']       = (X_feat['n_alarms'] > 0).astype(int)
        # interactions / disagreement / weighted sum
        X_feat['apri_x_fib4']  = X_norm['APRI']  * X_norm['FIB4']
        X_feat['apri_x_ritis'] = X_norm['APRI']  * X_norm['RITIS']
        X_feat['fib4_x_nafld'] = X_norm['FIB4'] * X_norm['NAFLD']
        X_feat['disagreement'] = X_norm.std(axis=1)
        X_feat['weighted_sum'] = (
            0.35*X_norm['APRI'] + 0.35*X_norm['FIB4'] +
            0.15*X_norm['RITIS'] + 0.15*X_norm['NAFLD']
        )
        # ratio-based
        X_feat['apri_over_fib4']  = self._safe_div(X['APRI'], X['FIB4'])
        X_feat['fib4_over_apri']  = self._safe_div(X['FIB4'], X['APRI'])
        X_feat['apri_over_ritis'] = self._safe_div(X['APRI'], X['RITIS'])
        X_feat['fib4_over_ritis'] = self._safe_div(X['FIB4'], X['RITIS'])
        X_feat['max_over_min']    = self._safe_div(X_feat['max_score'], X_feat['min_score'])
        X_feat['weighted_over_max'] = self._safe_div(X_feat['weighted_sum'], X_feat['max_score'])

        X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)  # imputer will handle the rest
        return X_feat

    # ---------- Threshold selection ----------

    def _find_dual_thresholds(self, y_true, y_proba):
        """Returns (T_low, T_high) after meeting the target conditions."""
        proba = np.clip(y_proba.astype(float), self.clip_eps, 1.0 - self.clip_eps)
        thr = np.unique(proba); thr.sort()

        def counts_at(t):
            yhat = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
            return tp, fp, tn, fn

        # --- T_low: LARGEST threshold with recall ≥ target_recall_low, NPV ≥ target_npv_low and min RULE-OUT
        T_low = None
        for t in thr[::-1]:  # from largest
            tp, fp, tn, fn = counts_at(t)
            prec, rec, spec, f1, ppv, npv = _metrics_from_counts(tp, fp, tn, fn)
            frac_ruleout = float(np.mean(proba < t))
            if (rec >= self.target_recall_low) and (npv >= self.target_npv_low) and (frac_ruleout >= self.min_ruleout_frac):
                T_low = float(t); break

        # fallback: F-beta (β=2) + min RULE-OUT
        if T_low is None:
            best_t, best_f = None, -1.0
            for t in thr[::-1]:
                tp, fp, tn, fn = counts_at(t)
                prec, rec, *_ = _metrics_from_counts(tp, fp, tn, fn)
                f = _fbeta(prec, rec, beta=self.beta_low)
                frac_ruleout = float(np.mean(proba < t))
                if (frac_ruleout >= self.min_ruleout_frac) and (f > best_f):
                    best_f, best_t = f, float(t)
            T_low = best_t  # may remain None in extreme datasets

        # --- T_high: SMALLEST threshold with spec ≥ target_spec and min RULE-IN
        T_high = None
        for t in thr:
            tp, fp, tn, fn = counts_at(t)
            prec, rec, spec, f1, ppv, npv = _metrics_from_counts(tp, fp, tn, fn)
            frac_rulein = float(np.mean(proba >= t))
            if (spec >= self.target_specificity_high) and (frac_rulein >= self.min_rulein_frac):
                T_high = float(t); break

        return T_low, T_high

    # ---------- Ensemble helpers ----------

    def _fit_base_estimators(self):
        gb = GradientBoostingClassifier(**self.gb_params)
        rf = RandomForestClassifier(**self.rf_params)
        if HAS_XGB:
            xgb = XGBClassifier(**self.xgb_params)
            return [('gb', gb), ('rf', rf), ('xgb', xgb)]
        else:
            return [('gb', gb), ('rf', rf)]

    def _fit_and_oof_raw(self, X, y, pos_weight):
        """OOF raw ensemble probabilities (uncalibrated)."""
        sw = compute_sample_weight(class_weight={0:1, 1:pos_weight}, y=y)
        skf = StratifiedKFold(n_splits=self.n_splits_oof, shuffle=True, random_state=self.random_state)
        y_oof = np.zeros_like(y, dtype=float)

        for tr, va in skf.split(X, y):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]
            w_tr = sw[tr].astype(np.float32)

            X_tr_f = self.create_advanced_features(X_tr)
            X_va_f = self.create_advanced_features(X_va)
            imp = SimpleImputer(strategy='median')
            X_tr_u = imp.fit_transform(X_tr_f).astype(np.float32)
            X_va_u = imp.transform(X_va_f).astype(np.float32)

            ests = self._fit_base_estimators()
            preds, used_w = [], []

            for i, (_, est) in enumerate(ests):
                est_clone = clone(est)
                est_clone.fit(X_tr_u, y_tr, sample_weight=w_tr)
                proba = est_clone.predict_proba(X_va_u)[:, 1]
                preds.append(np.clip(proba, self.clip_eps, 1 - self.clip_eps))
                used_w.append(self.ensemble_weights[i])

            used_w = np.array(used_w, dtype=float)
            used_w /= used_w.sum()
            y_oof[va] = np.clip(np.average(np.vstack(preds), axis=0, weights=used_w),
                                self.clip_eps, 1 - self.clip_eps)

        return y_oof, sw

    def _fit_full_models(self, X, y, pos_weight):
        """Final base models + imputer (uncalibrated) for the chosen configuration."""
        sw = compute_sample_weight(class_weight={0:1, 1:pos_weight}, y=y)
        X_f = self.create_advanced_features(X)
        self.imputer = SimpleImputer(strategy='median')
        X_u = self.imputer.fit_transform(X_f).astype(np.float32)

        ests = self._fit_base_estimators()
        models, used_w = [], []
        for i, (_, est) in enumerate(ests):
            est_clone = clone(est)
            est_clone.fit(X_u, y, sample_weight=sw)
            models.append(est_clone)
            used_w.append(self.ensemble_weights[i])

        used_w = np.array(used_w, dtype=float)
        used_w /= used_w.sum()
        self.models_ = models
        self.model_weights_ = used_w.tolist()

    # ---------- API fit/predict ----------

    def fit(self, X, y):
        best = None; best_info = None

        for w in self.weight_grid:
            # 1) raw OOF
            y_oof_raw, sw = self._fit_and_oof_raw(X, y, pos_weight=w)

            # 2) META-calibration on OOF
            if self.meta_calibration == "isotonic":
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(y_oof_raw, y)
                y_oof = np.clip(iso.predict(y_oof_raw), self.clip_eps, 1 - self.clip_eps)
                meta_cal = iso
            else:
                y_oof = y_oof_raw
                meta_cal = None

            # 3) thresholds + metrics after calibration
            T_low, T_high = self._find_dual_thresholds(y, y_oof)
            ap    = average_precision_score(y, y_oof)
            brier = brier_score_loss(y, y_oof)
            ece   = _ece_score(y, y_oof)

            # preference: both thresholds exist, then calibration quality
            score_tuple = (int(T_low is not None and T_high is not None), ap, -brier, -ece)
            if (best is None) or (score_tuple > best):
                best = score_tuple
                best_info = (w, y_oof, sw, T_low, T_high, ap, brier, ece, meta_cal)

        (self.best_weight, y_oof, self.sample_weights,
         self.threshold_low, self.threshold_high,
         ap, brier, ece, self.meta_calibrator_) = best_info

        self.cutoff_low  = None if self.threshold_low  is None else round(self.threshold_low*10, 2)
        self.cutoff_high = None if self.threshold_high is not None else None
        if self.threshold_high is not None:
            self.cutoff_high = round(self.threshold_high*10, 2)

        print("\n=== CONFIGURATION SELECTION (dual thresholds, META-calibration) ===")
        print(f"  Class-1 weight: {self.best_weight}")
        print(f"  OOF PR-AUC: {ap:.4f} | Brier: {brier:.4f} | ECE(10): {ece:.4f}")
        print(f"  T_low  (rec≥{self.target_recall_low}, NPV≥{self.target_npv_low}): {self.threshold_low} (cutoff {self.cutoff_low})")
        print(f"  T_high (spec≥{self.target_specificity_high}): {self.threshold_high} (cutoff {self.cutoff_high})")

        # 4) Final ensemble for this configuration
        self._fit_full_models(X, y, pos_weight=self.best_weight)
        return self

    def _proba_raw(self, X):
        Xf = self.create_advanced_features(X)
        Xu = self.imputer.transform(Xf).astype(np.float32)
        probs = [mdl.predict_proba(Xu)[:, 1] for mdl in self.models_]
        probs = np.vstack(probs)
        return np.clip(np.average(probs, axis=0, weights=self.model_weights_), self.clip_eps, 1 - self.clip_eps)

    def _proba(self, X):
        p_raw = self._proba_raw(X)
        if self.meta_calibrator_ is not None:
            p_cal = self.meta_calibrator_.predict(p_raw)
            return np.clip(p_cal, self.clip_eps, 1 - self.clip_eps)
        return p_raw

    # ---------- Evaluation / plots ----------

    def _cm_and_metrics(self, y_true, y_proba, threshold):
        y_hat = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
        return (np.array([[tn, fp],[fn, tp]]),) + _metrics_from_counts(tp, fp, tn, fn)

    def evaluate(self, X, y, plot=True):
        y_proba = self._proba(X)

        auc   = roc_auc_score(y, y_proba)
        ap    = average_precision_score(y, y_proba)
        brier = brier_score_loss(y, y_proba)
        ece   = _ece_score(y, y_proba)

        out = {
            'auc': auc, 'pr_auc': ap, 'brier': brier, 'ece': ece,
            'threshold_low': self.threshold_low, 'threshold_high': self.threshold_high,
            'cutoff_low': self.cutoff_low, 'cutoff_high': self.cutoff_high
        }

        # Metrics for T_low (if present)
        if self.threshold_low is not None:
            (cm_low, prec_l, rec_l, spec_l, f1_l, ppv_l, npv_l) = self._cm_and_metrics(y, y_proba, self.threshold_low)
            out.update({'low_confusion_matrix': cm_low,
                        'low_precision': prec_l, 'low_recall': rec_l,
                        'low_specificity': spec_l, 'low_f1': f1_l,
                        'low_ppv': ppv_l, 'low_npv': npv_l})

        # Metrics for T_high
        if self.threshold_high is not None:
            (cm_high, prec_h, rec_h, spec_h, f1_h, ppv_h, npv_h) = self._cm_and_metrics(y, y_proba, self.threshold_high)
            out.update({'high_confusion_matrix': cm_high,
                        'high_precision': prec_h, 'high_recall': rec_h,
                        'high_specificity': spec_h, 'high_f1': f1_h,
                        'high_ppv': ppv_h, 'high_npv': npv_h})

        # Triage breakdown
        n_ruleout = int(np.sum(y_proba <  (self.threshold_low  if self.threshold_low  is not None else 0.0)))
        n_rulein  = int(np.sum(y_proba >= (self.threshold_high if self.threshold_high is not None else 1.0)))
        n_gray    = int(len(y_proba) - n_ruleout - n_rulein)
        out.update({'n_ruleout': n_ruleout, 'n_gray': n_gray, 'n_rulein': n_rulein})

        # Compact printout
        print("\n" + "="*60)
        print("RESULTS (dual thresholds, META-calibration) — FSCORE 1 vs 2–4")
        print("="*60)
        print(f"AUC-ROC: {auc:.4f} | PR-AUC: {ap:.4f} | Brier: {brier:.4f} | ECE(10): {ece:.4f}")
        print(f"T_low : {self.threshold_low}  (cutoff {self.cutoff_low})")
        if self.threshold_low is not None:
            print(f"  → recall={out['low_recall']:.3f} | specificity={out['low_specificity']:.3f} | "
                  f"precision={out['low_precision']:.3f} | NPV={out['low_npv']:.3f} | F1={out['low_f1']:.3f}")
        print(f"T_high: {self.threshold_high} (cutoff {self.cutoff_high})")
        if self.threshold_high is not None:
            print(f"  → recall={out['high_recall']:.3f} | specificity={out['high_specificity']:.3f} | "
                  f"precision={out['high_precision']:.3f} | PPV={out['high_ppv']:.3f} | F1={out['high_f1']:.3f}")
        print(f"Triage: rule-out={out['n_ruleout']} | gray={out['n_gray']} | rule-in={out['n_rulein']}")

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 1) Confusion matrix (T_low, if exists)
            if self.threshold_low is not None:
                cm = out['low_confusion_matrix']
                title = f"Confusion matrix (T_low = {self.cutoff_low})"
                p, r, s, f1 = out['low_precision'], out['low_recall'], out['low_specificity'], out['low_f1']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                            xticklabels=['FSCORE 1\n(Predicted)', 'FSCORE 2–4\n(Predicted)'],
                            yticklabels=['FSCORE 1\n(Actual)', 'FSCORE 2–4\n(Actual)'])
                axes[0,0].set_title(title)
                axes[0,0].text(0.5, -0.18,
                    f"precision={p:.3f} | recall={r:.3f} | specificity={s:.3f} | F1={f1:.3f}",
                    ha='center', va='center', transform=axes[0,0].transAxes)
            else:
                axes[0,0].axis('off')

            # 2) ROC curve
            fpr, tpr, _ = roc_curve(y, y_proba)
            axes[0,1].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
            axes[0,1].plot([0,1], [0,1], 'k--', alpha=0.5)
            axes[0,1].fill_between(fpr, tpr, alpha=0.3)
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate (Recall)')
            axes[0,1].set_title('ROC Curve (FSCORE 1 vs 2–4)')
            axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

            # 3) Score distribution with both thresholds
            scores_h = y_proba[y == 0] * 10
            scores_s = y_proba[y == 1] * 10
            axes[1,0].hist(scores_h, bins=30, alpha=0.7, label='FSCORE 1', color='blue', edgecolor='black')
            axes[1,0].hist(scores_s, bins=30, alpha=0.7, label='FSCORE 2–4', color='red', edgecolor='black')
            if self.cutoff_low is not None:
                axes[1,0].axvline(self.cutoff_low, color='black', linestyle='--', linewidth=2, label=f'T_low = {self.cutoff_low}')
            if self.cutoff_high is not None:
                axes[1,0].axvline(self.cutoff_high, color='black', linestyle='-.', linewidth=2, label=f'T_high = {self.cutoff_high}')
            axes[1,0].set_xlabel('Meta-calculator score (0–10)')
            axes[1,0].set_ylabel('Number of patients')
            axes[1,0].set_title('Score distribution & thresholds (≥F2)')
            axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

            # 4) Confusion matrix (T_high)
            if self.threshold_high is not None:
                cm = out['high_confusion_matrix']
                title = f"Confusion matrix (T_high = {self.cutoff_high})"
                p, r, s, f1 = out['high_precision'], out['high_recall'], out['high_specificity'], out['high_f1']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
                            xticklabels=['FSCORE 1\n(Predicted)', 'FSCORE 2–4\n(Predicted)'],
                            yticklabels=['FSCORE 1\n(Actual)', 'FSCORE 2–4\n(Actual)'])
                axes[1,1].set_title(title)
                axes[1,1].text(0.5, -0.14,
                    f"precision={p:.3f} | recall={r:.3f} | specificity={s:.3f} | F1={f1:.3f}",
                    ha='center', va='center', transform=axes[1,1].transAxes)
            else:
                axes[1,1].axis('off')

            plt.tight_layout(); plt.show()

        return out

    # ---------- Single study ----------

    def calculate_score(self, apri, fib4, ritis, nafld):
        """Return score and zone for the 1 vs 2–4 (≥F2) classification."""
        X = pd.DataFrame([[apri, fib4, ritis, nafld]], columns=['APRI', 'FIB4', 'RITIS', 'NAFLD'])
        proba = float(self._proba(X)[0])
        score = round(proba * 10, 2)
        zone = 'GRAY'
        if (self.threshold_low is not None) and (proba < self.threshold_low):
            zone = 'RULE_OUT'
        elif (self.threshold_high is not None) and (proba >= self.threshold_high):
            zone = 'RULE_IN'
        return {'probability': round(proba, 3), 'score_0_10': score, 'zone': zone,
                'T_low': self.cutoff_low, 'T_high': self.cutoff_high}


# ========= Distillation: closed-form logit-linear + export & fidelity =========
import os
import json
from sklearn.linear_model import LinearRegression

class LinearFormulaDistiller:
    """
    Distill the ensemble into a closed-form:
        logit(p) = beta0 + sum_j beta_j * x_j
        p = 1 / (1 + exp(-logit(p)))
        score_0_10 = 10 * p

    Note: x_j are features produced by your create_advanced_features().
    """
    def __init__(self, feature_names, clip_eps=1e-6):
        self.feature_names = list(feature_names)
        self.clip_eps = clip_eps
        self.imputer = SimpleImputer(strategy='median')
        self.lr = LinearRegression()
        self.coef_ = None
        self.intercept_ = None
        self.medians_ = None  # for imputing NaNs at inference

    @staticmethod
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, calculator, X_ref: pd.DataFrame):
        """
        calculator: trained LiverFibrosisMetaCalculatorF2plusDual
        X_ref:      reference data for distillation (e.g., X_tr)
        """
        # 1) features
        X_feat = calculator.create_advanced_features(X_ref)
        # sanity: ensure all features are present
        missing = [c for c in self.feature_names if c not in X_feat.columns]
        if missing:
            raise ValueError(f"Missing features in create_advanced_features(): {missing}")
        X_sel = X_feat[self.feature_names].copy()

        # 2) median imputation
        X_imp = self.imputer.fit_transform(X_sel)
        self.medians_ = pd.Series(self.imputer.statistics_, index=self.feature_names)

        # 3) teacher targets: logits of calibrated ensemble
        p_teacher = calculator._proba(X_ref)
        y_logit = self._logit(p_teacher, self.clip_eps)

        # 4) linear regression on logits
        self.lr.fit(X_imp, y_logit)
        self.coef_ = pd.Series(self.lr.coef_, index=self.feature_names)
        self.intercept_ = float(self.lr.intercept_)
        return self

    def predict_proba_from_features(self, X_features: pd.DataFrame) -> np.ndarray:
        """Predict p from already-computed features (create_advanced_features)."""
        X_sel = X_features[self.feature_names].copy()
        # fill any NaNs with learned medians
        for c in self.feature_names:
            X_sel[c] = X_sel[c].fillna(self.medians_[c])
        z = self.intercept_ + np.dot(X_sel.values, self.coef_.values)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p, self.clip_eps, 1 - self.clip_eps)

    # ----- export -----

    def export_dict(self):
        return {
            "intercept": self.intercept_,
            "coefficients": {k: float(v) for k, v in self.coef_.items()},
            "feature_order": self.feature_names,
            "impute_medians": {k: float(v) for k, v in self.medians_.items()},
            "formulas": {
                "logit": "logit(p) = intercept + sum_j coef[j] * x_j",
                "prob": "p = 1 / (1 + exp(-logit(p)))",
                "score_0_10": "score = 10 * p"
            }
        }

    def export_text(self, cutoff_high=None, cutoff_low=None):
        lines = []
        lines.append("Closed-form meta-calculator (distilled from ensemble):")
        lines.append(f"logit(p) = {self.intercept_:.6f}")
        for k, v in self.coef_.items():
            lines.append(f"           + ({v:+.6f}) * {k}")
        lines.append("p = 1 / (1 + exp(-logit(p)))")
        lines.append("Score_0-10 = 10 * p")
        if cutoff_low is not None:
            lines.append(f"T_low  (prob) = {cutoff_low:.6f}  |  T_low (score) = {cutoff_low*10:.2f}")
        if cutoff_high is not None:
            lines.append(f"T_high (prob) = {cutoff_high:.6f} |  T_high (score) = {cutoff_high*10:.2f}")
        return "\n".join(lines)

    def export_latex(self, var_map=None):
        def nm(x):
            return var_map.get(x, x) if var_map else x
        terms = [f"{self.intercept_:.6f}"]
        for k, v in self.coef_.items():
            terms.append(f"{v:+.6f}\\cdot {nm(k)}")
        logit_expr = " ".join([" +".join(terms[:1]), "+ " + " + ".join(terms[1:])]) if len(terms) > 1 else terms[0]
        latex = (
            "\\begin{aligned}\n"
            "\\text{logit}(p) &= " + logit_expr + " \\\\\n"
            "p &= \\frac{1}{1 + e^{-\\text{logit}(p)}} \\\\\n"
            "\\text{Score}_{0\\text{--}10} &= 10\\cdot p\n"
            "\\end{aligned}\n"
        )
        return latex


def distill_and_save_formula(
    calculator,
    X_ref: pd.DataFrame,
    features: list,
    out_dir: str = "artifacts",
    tag: str = "f2plus_dual"
):
    """
    Trains the formula, saves JSON/TXT/TEX and returns the distiller.
    """
    os.makedirs(out_dir, exist_ok=True)
    dist = LinearFormulaDistiller(feature_names=features)
    dist.fit(calculator, X_ref)

    # saves
    txt = dist.export_text(
        cutoff_high=calculator.threshold_high,
        cutoff_low=calculator.threshold_low
    )
    with open(os.path.join(out_dir, f"formula_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    with open(os.path.join(out_dir, f"formula_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dist.export_dict(), f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, f"formula_{tag}.tex"), "w", encoding="utf-8") as f:
        f.write(dist.export_latex())

    print("\n[Distillation] Saved:")
    print(f"  - {os.path.join(out_dir, f'formula_{tag}.txt')}")
    print(f"  - {os.path.join(out_dir, f'formula_{tag}.json')}")
    print(f"  - {os.path.join(out_dir, f'formula_{tag}.tex')}")
    return dist


def evaluate_formula_fidelity(
    calculator,
    distiller: LinearFormulaDistiller,
    X: pd.DataFrame, y: np.ndarray,
    threshold: float = None,
    title: str = "Formula fidelity vs Ensemble"
):
    """
    Compares p_formula to p_teacher and (optionally) decision metrics at the same threshold.
    """
    # teacher
    p_teacher = calculator._proba(X)
    # student / formula
    X_feat = calculator.create_advanced_features(X)
    p_formula = distiller.predict_proba_from_features(X_feat)

    # global metrics
    auc_t  = roc_auc_score(y, p_teacher)
    auc_f  = roc_auc_score(y, p_formula)
    ap_t   = average_precision_score(y, p_teacher)
    ap_f   = average_precision_score(y, p_formula)
    brier_t = brier_score_loss(y, p_teacher)
    brier_f = brier_score_loss(y, p_formula)
    ece_t   = _ece_score(y, p_teacher)
    ece_f   = _ece_score(y, p_formula)

    corr   = np.corrcoef(p_teacher, p_formula)[0,1]
    mae    = float(np.mean(np.abs(p_teacher - p_formula)))

    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"Corr(p_teacher, p_formula): {corr:.4f} | MAE: {mae:.4f}")
    print(f"AUC  teacher={auc_t:.4f} | formula={auc_f:.4f}")
    print(f"PR-AUC teacher={ap_t:.4f} | formula={ap_f:.4f}")
    print(f"Brier teacher={brier_t:.4f} | formula={brier_f:.4f}")
    print(f"ECE(10) teacher={ece_t:.4f} | formula={ece_f:.4f}")

    # decision metrics at the same threshold (e.g., T_high)
    if threshold is not None:
        def cm_and_metrics(p):
            y_hat = (p >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0,1]).ravel()
            prec, rec, spec, f1, ppv, npv = _metrics_from_counts(tp, fp, tn, fn)
            return (tn, fp, fn, tp, prec, rec, spec, f1, ppv, npv)

        tn, fp, fn, tp, pr, rc, sp, f1, ppv, npv = cm_and_metrics(p_teacher)
        print(f"\nDecision metrics @ threshold={threshold:.6f} (teacher):")
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp} | prec={pr:.3f} rec={rc:.3f} spec={sp:.3f} F1={f1:.3f} PPV={ppv:.3f} NPV={npv:.3f}")

        tn, fp, fn, tp, pr, rc, sp, f1, ppv, npv = cm_and_metrics(p_formula)
        print(f"Decision metrics @ threshold={threshold:.6f} (formula):")
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp} | prec={pr:.3f} rec={rc:.3f} spec={sp:.3f} F1={f1:.3f} PPV={ppv:.3f} NPV={npv:.3f}")

    return {
        "corr": float(corr), "mae": mae,
        "auc_teacher": float(auc_t), "auc_formula": float(auc_f),
        "ap_teacher": float(ap_t), "ap_formula": float(ap_f),
        "brier_teacher": float(brier_t), "brier_formula": float(brier_f),
        "ece_teacher": float(ece_t), "ece_formula": float(ece_f)
    }



# ================== Pipeline ==================

def main_pipeline(csv_path, do_plots=True):
    print("="*60)
    print("HEPATOLOGY META-CALCULATOR — FSCORE 1 vs 2–4 (Dual thresholds)")
    print("Goal: screening (rule-out) + confirmation (rule-in) of significant fibrosis (≥F2)")
    print("="*60)

    print("\n1. Loading data...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} records")

    print("\n2. Preparing data...")
    calc = LiverFibrosisMetaCalculatorF2plusDual(
        target_recall_low=0.95,
        target_npv_low=0.95,
        target_specificity_high=0.80,
        beta_low=2.0,
        weight_grid=(5, 10, 15),
        n_splits_oof=5,
        random_state=42,
        min_ruleout_frac=0.10,
        min_rulein_frac=0.05,
        ensemble_weights=(0.5, 0.35, 0.15),
        meta_calibration="isotonic",
    )
    X, y = calc.prepare_data(df)
    print(f"   FSCORE=1:  {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   FSCORE=2–4:{(y == 1).sum()} ({(y == 1).mean():.1%})")

    print("\n3. Train/test split...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=41, stratify=y)
    print(f"   Train set: {len(X_tr)} samples")
    print(f"   Test set:  {len(X_te)} samples")

    print("\n4. Training & configuration selection...")
    calc.fit(X_tr, y_tr)
    print("   ✓ Model trained, meta-calibrated, thresholds selected")

    print("\n5. Evaluation on the test set:")
    metrics = calc.evaluate(X_te, y_te, plot=do_plots)
    # === Distillation of a formula (LIGHT and HIFI versions) ===
    FEATURES_LIGHT = ["APRI", "FIB4", "RITIS", "NAFLD", "apri_x_fib4", "apri_x_ritis"]
    FEATURES_HIFI = ['max_score', 'APRI', 'FIB4', 'weighted_sum', 'apri_x_ritis', 'RITIS',
                     'apri_x_fib4', 'min_score', 'std_score', 'fib4_x_nafld']

    dist_light = distill_and_save_formula(calc, X_tr, FEATURES_LIGHT, out_dir="artifacts", tag="f2_dual_light")
    evaluate_formula_fidelity(calc, dist_light, X_te, y_te, threshold=calc.threshold_high,
                              title="Formula (LIGHT) fidelity vs Ensemble")

    dist_hifi = distill_and_save_formula(calc, X_tr, FEATURES_HIFI, out_dir="artifacts", tag="f2_dual_hifi")
    evaluate_formula_fidelity(calc, dist_hifi, X_te, y_te, threshold=calc.threshold_high,
                              title="Formula (HIFI) fidelity vs Ensemble")

    print("\n6. Saving the model...")
    joblib.dump(calc, 'liver_fibrosis_meta_calibrator_F2_dual.pkl')
    print("   ✓ Saved: 'liver_fibrosis_meta_calibrator_F2_dual.pkl'")

    return calc, metrics


if __name__ == "__main__":
    # Path to CSV with FSCORE ∈ {1,2,3,4}
    csv_path = "Data//Modified//Full//Fscore-No-Cat-Imp-Full.csv"
    calculator, metrics = main_pipeline(csv_path, do_plots=True)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Class-1 weight: {calculator.best_weight}")
    print(f"✓ T_low : {calculator.threshold_low}  (cutoff {calculator.cutoff_low})")
    print(f"✓ T_high: {calculator.threshold_high} (cutoff {calculator.cutoff_high})")
    print(f"✓ Test AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"✓ Test Brier: {metrics['brier']:.4f} | ECE(10): {metrics['ece']:.4f}")
