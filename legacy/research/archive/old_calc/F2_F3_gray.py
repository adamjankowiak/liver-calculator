
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

    def __init__(
        self,
        target_recall_low=0.95,             # rule-out (sensitivity) target
        target_npv_low=0.95,                # rule-out NPV target
        target_specificity_high=0.80,       # rule-in (specificity) target
        beta_low=2.0,                       # F-beta fallback for T_low
        weight_grid=(1,2,3,4,5,6,7,8,9, 10,15,20,25,30),            # positive-class weights (≥F2)
        n_splits_oof=5,
        random_state=42,
        min_ruleout_frac=0.10,              # minimum fraction sent to RULE_OUT (e.g., ≥10%)
        min_rulein_frac=0.05,               # minimum fraction sent to RULE_IN
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

        # Base estimators (no calibration here):
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

        # Fitted state
        self.best_weight = None
        self.sample_weights = None

        self.imputer = SimpleImputer(strategy='median')   # set on full-train
        self.models_ = []                                 # list of base models
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
                raise ValueError(f"Missing column: {c}")

        fs = pd.to_numeric(df['FSCORE'], errors='coerce')
        bad = fs.isna().sum()
        if bad:
            raise ValueError(f"FSCORE contains {bad} unparsable values (expected 1/2/3/4).")
        unique_vals = np.sort(fs.unique())

        # target: 0 for 1–2, 1 for 3–4  (grayzone ma sens klinicznie między 2 i 3)
        y = np.where(fs <= 2, 0, 1).astype(int)
        X = df[['APRI', 'FIB4', 'RITIS', 'NAFLD']].copy()

        print(f"   Unique FSCORE values: {unique_vals}")
        print("   Mapping: 1–2 → 0 (≤F2), 3–4 → 1 (≥F3)")
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
        X_feat['apri_alarm']  = (X['APRI']  >= 1.0).astype(int)
        X_feat['fib4_alarm']  = (X['FIB4']  >= 3.25).astype(int)
        X_feat['ritis_alarm'] = (X['RITIS'] >= 1.0).astype(int)
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

        X_feat.replace([np.inf, -np.inf], np.nan, inplace=True)  # imputer handles the rest
        return X_feat

    # ---------- Threshold search ----------

    def _find_dual_thresholds(self, y_true, y_proba):
        """Return (T_low, T_high) that satisfy the target conditions."""
        proba = np.clip(y_proba.astype(float), self.clip_eps, 1.0 - self.clip_eps)
        thr = np.unique(proba); thr.sort()

        def counts_at(t):
            yhat = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
            return tp, fp, tn, fn

        # --- T_low: LARGEST threshold with recall ≥ target, NPV ≥ target, and min RULE_OUT
        T_low = None
        for t in thr[::-1]:
            tp, fp, tn, fn = counts_at(t)
            prec, rec, spec, f1, ppv, npv = _metrics_from_counts(tp, fp, tn, fn)
            frac_ruleout = float(np.mean(proba < t))
            if (rec >= self.target_recall_low) and (npv >= self.target_npv_low) and (frac_ruleout >= self.min_ruleout_frac):
                T_low = float(t); break

        # fallback: F-beta (β=2) + min RULE_OUT
        if T_low is None:
            best_t, best_f = None, -1.0
            for t in thr[::-1]:
                tp, fp, tn, fn = counts_at(t)
                prec, rec, *_ = _metrics_from_counts(tp, fp, tn, fn)
                f = _fbeta(prec, rec, beta=self.beta_low)
                frac_ruleout = float(np.mean(proba < t))
                if (frac_ruleout >= self.min_ruleout_frac) and (f > best_f):
                    best_f, best_t = f, float(t)
            T_low = best_t  # may remain None on extreme data

        # --- T_high: SMALLEST threshold with spec ≥ target and min RULE_IN
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
        """OOF raw ensemble probabilities (no calibration)."""
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
        """Final base models + imputer (no calibration) for the selected config."""
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
            # 1) OOF raw
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

            # 3) Thresholds + metrics after calibration
            T_low, T_high = self._find_dual_thresholds(y, y_oof)
            ap    = average_precision_score(y, y_oof)
            brier = brier_score_loss(y, y_oof)
            ece   = _ece_score(y, y_oof)

            # preference: both thresholds found, then calibration quality
            score_tuple = (int(T_low is not None and T_high is not None), ap, -brier, -ece)
            if (best is None) or (score_tuple > best):
                best = score_tuple
                best_info = (w, y_oof, sw, T_low, T_high, ap, brier, ece, meta_cal)

        (self.best_weight, y_oof, self.sample_weights,
         self.threshold_low, self.threshold_high,
         ap, brier, ece, self.meta_calibrator_) = best_info

        self.cutoff_low  = None if self.threshold_low  is None else round(self.threshold_low*10, 2)
        self.cutoff_high = None if self.threshold_high is None else round(self.threshold_high*10, 2)

        print("\n=== CONFIGURATION SELECTION (dual thresholds, META-calibration) ===")
        print(f"  Positive-class weight: {self.best_weight}")
        print(f"  OOF PR-AUC: {ap:.4f} | Brier: {brier:.4f} | ECE(10): {ece:.4f}")
        print(f"  T_low  (rec≥{self.target_recall_low}, NPV≥{self.target_npv_low}): {self.threshold_low} (cutoff {self.cutoff_low})")
        print(f"  T_high (spec≥{self.target_specificity_high}): {self.threshold_high} (cutoff {self.cutoff_high})")

        # 4) Final ensemble for the selected configuration
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

        # Triage counts
        n_ruleout = int(np.sum(y_proba <  (self.threshold_low  if self.threshold_low  is not None else 0.0)))
        n_rulein  = int(np.sum(y_proba >= (self.threshold_high if self.threshold_high is not None else 1.0)))
        n_gray    = int(len(y_proba) - n_ruleout - n_rulein)
        out.update({'n_ruleout': n_ruleout, 'n_gray': n_gray, 'n_rulein': n_rulein})

        # Console report
        print("\n" + "="*60)
        print("RESULTS (dual thresholds, META-calibration) — FSCORE 1–2 vs 3–4")
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
        print(f"Triage: rule-out={out['n_ruleout']} | gray-zone={out['n_gray']} | rule-in={out['n_rulein']}")

        if plot:
            # 1) stabilny layout
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

            # --- A) Confusion matrix (T_low)
            if self.threshold_low is not None:
                cm = out['low_confusion_matrix']
                title = f"Confusion matrix (T_low = {self.cutoff_low})"
                p, r, s, f1 = out['low_precision'], out['low_recall'], out['low_specificity'], out['low_f1']
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Predicted FSCORE 1', 'Predicted FSCORE 2–4'],
                    yticklabels=['Actual FSCORE 1', 'Actual FSCORE 2–4'],
                    cbar_kws={'shrink': 0.85}
                )
                axes[0, 0].set_title(title, pad=12)
                axes[0, 0].text(
                    0.5, -0.22,
                    f"precision={p:.3f} | recall={r:.3f} | specificity={s:.3f} | F1={f1:.3f}",
                    ha='center', va='center', transform=axes[0, 0].transAxes, clip_on=False
                )
            else:
                axes[0, 0].axis('off')

            # --- B) ROC
            fpr, tpr, _ = roc_curve(y, y_proba)
            axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 1].fill_between(fpr, tpr, alpha=0.3)
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate (Recall)')
            axes[0, 1].set_title('ROC curve (FSCORE 1–2 vs 3–4)', pad=12)
            axes[0, 1].legend(loc='lower right', frameon=False)
            axes[0, 1].grid(True, alpha=0.3)

            # --- C) Score distribution + thresholds
            scores_h = y_proba[y == 0] * 10
            scores_s = y_proba[y == 1] * 10
            axes[1, 0].hist(scores_h, bins=30, alpha=0.7, label='FSCORE 1–2', color='blue', edgecolor='black')
            axes[1, 0].hist(scores_s, bins=30, alpha=0.7, label='FSCORE 3–4', color='red', edgecolor='black')
            if self.cutoff_low is not None:
                axes[1, 0].axvline(self.cutoff_low, color='black', linestyle='--', linewidth=2,
                                   label=f'T_low = {self.cutoff_low}')
            if self.cutoff_high is not None:
                axes[1, 0].axvline(self.cutoff_high, color='black', linestyle='-.', linewidth=2,
                                   label=f'T_high = {self.cutoff_high}')
            axes[1, 0].set_xlabel('Meta-calculator score (0–10)')
            axes[1, 0].set_ylabel('Number of patients')
            axes[1, 0].set_title('Score distribution & thresholds (≥F2)', pad=12)
            # przenieś legendę poza wykres, żeby nie kolidowała
            axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., frameon=False)
            fig.subplots_adjust(right=0.88)  # miejsce na legendę po prawej
            axes[1, 0].grid(True, alpha=0.3)

            # --- D) Confusion matrix (T_high)
            if self.threshold_high is not None:
                cm = out['high_confusion_matrix']
                title = f"Confusion matrix (T_high = {self.cutoff_high})"
                p, r, s, f1 = out['high_precision'], out['high_recall'], out['high_specificity'], out['high_f1']
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                    xticklabels=['Predicted FSCORE 1–2', 'Predicted FSCORE 3–4'],
                    yticklabels=['Actual FSCORE 1–2', 'Actual FSCORE 3–4'],
                    cbar_kws={'shrink': 0.85}
                )
                axes[1, 1].set_title(title, pad=12)
                axes[1, 1].text(
                    0.5, -0.20,
                    f"precision={p:.3f} | recall={r:.3f} | specificity={s:.3f} | F1={f1:.3f}",
                    ha='center', va='center', transform=axes[1, 1].transAxes, clip_on=False
                )
            else:
                axes[1, 1].axis('off')

            # 2) zapis *po* ułożeniu layoutu
            # (jeśli nie używasz constrained_layout: odkomentuj następne dwie linie)
            # fig.tight_layout()
            # fig.canvas.draw()  # upewnia rozmieszczenie textów przed zapisem
            fig.savefig('F2_F3_reduced_iterative.png', dpi=400, bbox_inches='tight')
            plt.show()

        return out

    # ---------- Single patient ----------

    def calculate_score(self, apri, fib4, ritis, nafld):
        """Return score and triage zone for the 1-2 vs 3–4 (≥F3) task."""
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



# ================== Pipeline ==================

def main_pipeline(csv_path, do_plots=True):
    print("="*60)
    print("HEPATOLOGY META-CALCULATOR — FSCORE 1–2 vs 3–4 (Dual thresholds)")
    print("Goal: rule-out ≤F2 + rule-in ≥F3 (gray zone around boundary 2 vs 3)")
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
        weight_grid=(1,2,3,4,5,6,7,8,9, 10,15, 20,25,30),
        n_splits_oof=5,
        random_state=40,
        min_ruleout_frac=0.10,
        min_rulein_frac=0.05,
        ensemble_weights=(0.5, 0.35, 0.15),
        meta_calibration="isotonic",
    )
    X, y = calc.prepare_data(df)
    print(f"   FSCORE=1–2: {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   FSCORE=3–4: {(y == 1).sum()} ({(y == 1).mean():.1%})")

    print("\n3. Train/test split...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"   Train set: {len(X_tr)} samples")
    print(f"   Test set:  {len(X_te)} samples")

    print("\n4. Training & configuration selection...")
    calc.fit(X_tr, y_tr)
    print("   ✓ Model trained, meta-calibrated, thresholds selected")

    print("\n5. Evaluation on the test set:")
    metrics = calc.evaluate(X_te, y_te, plot=do_plots)

    print("\n6. Saving the model...")
    joblib.dump(calc, 'liver_fibrosis_meta_calibrator_F3_dual.pkl')
    print("   ✓ Saved: 'liver_fibrosis_meta_calibrator_F2_dual.pkl'")

    return calc, metrics


if __name__ == "__main__":
    # Path to CSV with FSCORE ∈ {1,2,3,4}
    csv_path = "Data/Modified/Full/Final/Reduced-Imp-Iterative.csv"
    calculator, metrics = main_pipeline(csv_path, do_plots=True)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Positive-class weight: {calculator.best_weight}")
    print(f"✓ T_low : {calculator.threshold_low}  (cutoff {calculator.cutoff_low})")
    print(f"✓ T_high: {calculator.threshold_high} (cutoff {calculator.cutoff_high})")
    print(f"✓ Test AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"✓ Test Brier: {metrics['brier']:.4f} | ECE(10): {metrics['ece']:.4f}")