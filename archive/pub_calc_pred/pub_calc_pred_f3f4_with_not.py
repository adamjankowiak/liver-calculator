import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd


# Progi (ściśle: > próg)
THRESHOLDS: Dict[str, float] = {
    "FIB4": 3.25,     # FIB-4: 1.45,grayzone,3.25
    "APRI": 1,      # APRI:  0.5,grayzone,1
    "NAFLD": 6.76,   # NAFLD: -1.455,grayzone,0.676
    "RITIS": 1.0,     # de Ritis Ratio: 1
}


@dataclass
class Metrics:
    tp: int
    fn: int
    fp: int
    tn: int
    precision: float
    recall: float
    specificity: float
    npv: float
    f1: float
    accuracy: float
    balanced_accuracy: float


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(((y_true == True) & (y_pred == True)).sum())
    fn = int(((y_true == True) & (y_pred == False)).sum())
    fp = int(((y_true == False) & (y_pred == True)).sum())
    tn = int(((y_true == False) & (y_pred == False)).sum())
    return tp, fn, fp, tn


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Metrics:
    tp, fn, fp, tn = confusion_counts(y_true, y_pred)

    precision = safe_div(tp, tp + fp)          # PPV
    recall = safe_div(tp, tp + fn)             # sensitivity
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2.0

    return Metrics(
        tp=tp, fn=fn, fp=fp, tn=tn,
        precision=precision, recall=recall, specificity=specificity,
        npv=npv, f1=f1, accuracy=accuracy, balanced_accuracy=balanced_accuracy
    )


def add_diagnostics(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    """
    Dodaje kolumny:
      - alarms_count: ile progów przekroczono (0..N)
      - alarms_list: lista nazw wskaźników, które przekroczyły próg
      - margin_<SCORE>: (wartość - próg) -> dodatnie = ponad próg, ujemne = poniżej
    """
    pred_cols = [f"pred_{c}" for c in score_cols]

    df["alarms_count"] = df[pred_cols].sum(axis=1).astype(int)

    def _alarms_list(row):
        hits = [c for c in score_cols if bool(row.get(f"pred_{c}", False))]
        return "|".join(hits) if hits else ""

    df["alarms_list"] = df.apply(_alarms_list, axis=1)

    for c in score_cols:
        thr = THRESHOLDS[c]
        x = pd.to_numeric(df[c], errors="coerce")
        df[f"margin_{c}"] = round(x - thr,2)

    return df


def save_disagreements(
    df: pd.DataFrame,
    y_true: pd.Series,
    pred_col: str,
    base_prefix: str,
    keep_cols: List[str] | None = None
) -> Tuple[str, str]:
    """
    Zapisuje dwa pliki:
      - <base_prefix>_FP.csv
      - <base_prefix>_FN.csv
    """
    y_pred = df[pred_col].astype(bool)
    y_true_b = y_true.astype(bool)

    fp_mask = (~y_true_b) & (y_pred)
    fn_mask = (y_true_b) & (~y_pred)

    fp_df = df.loc[fp_mask].copy()
    fn_df = df.loc[fn_mask].copy()

    # Ułatwienie: dołącz etykiety
    fp_df["y_true_F3F4"] = y_true_b.loc[fp_mask].values
    fp_df["y_pred_F3F4"] = y_pred.loc[fp_mask].values
    fn_df["y_true_F3F4"] = y_true_b.loc[fn_mask].values
    fn_df["y_pred_F3F4"] = y_pred.loc[fn_mask].values

    # (Opcjonalnie) zawęź kolumny
    if keep_cols is not None:
        # Zachowaj też kolumny predykcyjne i diagnostyczne jeśli istnieją
        extra = [pred_col, "pred_ANY_OR", "alarms_count", "alarms_list"]
        extra += [c for c in df.columns if c.startswith("pred_") or c.startswith("margin_")]
        cols = []
        for c in (keep_cols + extra):
            if c in df.columns and c not in cols:
                cols.append(c)
        fp_df = fp_df[cols]
        fn_df = fn_df[cols]

    fp_path = f"{base_prefix}_FP.csv"
    fn_path = f"{base_prefix}_FN.csv"
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)

    return fp_path, fn_path


def main():
    parser = argparse.ArgumentParser(description="Ocena progów FIB4/APRI/NAFLD/RITIS dla klasy F3/F4 (FSCORE='3-4').")
    parser.add_argument("--csv", default="Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv", help="Ścieżka do pliku CSV.")
    parser.add_argument("--out", default="Fscore-OR-predictions.csv", help="Gdzie zapisać CSV z predykcjami (pełny).")
    parser.add_argument("--disagreements-dir", default="disagreements", help="Folder na pliki FP/FN.")
    parser.add_argument("--by-rule", action="store_true",
                        help="Jeśli ustawisz, zapisze FP/FN osobno dla każdej reguły (FIB4/APRI/NAFLD/RITIS/ANY_OR).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = ["FSCORE"] + list(THRESHOLDS.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w CSV: {missing}. Dostępne kolumny: {list(df.columns)}")

    # Prawdziwa klasa: FSCORE == "3-4"
    y_true = df["FSCORE"].astype(str).str.strip().eq("3-4")

    # Klasa ujemna: "nie F3/F4"
    y_true_notF3F4 = ~y_true

    # Predykcje pojedynczych wskaźników
    score_cols = list(THRESHOLDS.keys())
    pred_cols = []
    for col, thr in THRESHOLDS.items():
        x = pd.to_numeric(df[col], errors="coerce")
        pred = (x > thr).fillna(False)  # NaN => brak przekroczenia progu
        pred_name = f"pred_{col}"
        df[pred_name] = pred
        pred_cols.append(pred_name)

    # Reguła OR: jeśli jakikolwiek przekroczył próg
    df["pred_ANY_OR"] = df[pred_cols].any(axis=1)

    # Diagnostyka (ile alarmów, które, marginesy)
    df = add_diagnostics(df, score_cols)

    # Raport metryk (pozytyw = F3/F4) + metryki dla klasy przeciwnej (pozytyw = nie F3/F4)
    rows = []
    for name in pred_cols + ["pred_ANY_OR"]:
        y_pred = df[name].astype(bool)

        m_pos = compute_metrics(y_true, y_pred)

        # Metryki dla "nie F3/F4" traktowane jako klasa pozytywna:
        # y_true_notF3F4 = ~y_true, y_pred_notF3F4 = ~y_pred
        m_neg = compute_metrics(y_true_notF3F4, ~y_pred)

        rows.append({
            "rule": name.replace("pred_", ""),

            # Pozytyw = F3/F4
            "TP": m_pos.tp, "FN": m_pos.fn, "FP": m_pos.fp, "TN": m_pos.tn,
            "precision(PPV)": round(m_pos.precision, 4),
            "recall(sensitivity)": round(m_pos.recall, 4),
            "specificity": round(m_pos.specificity, 4),
            "NPV": round(m_pos.npv, 4),
            "F1": round(m_pos.f1, 4),
            "accuracy": round(m_pos.accuracy, 4),
            "balanced_accuracy": round(m_pos.balanced_accuracy, 4),

            # Pozytyw = nie F3/F4 (czyli negacja)
            "precision_notF3F4": round(m_neg.precision, 4),
            "recall_notF3F4": round(m_neg.recall, 4),
            "specificity_notF3F4": round(m_neg.specificity, 4),
            "NPV_notF3F4": round(m_neg.npv, 4),
            "F1_notF3F4": round(m_neg.f1, 4),
            "balanced_accuracy_notF3F4": round(m_neg.balanced_accuracy, 4),
        })

    report = pd.DataFrame(rows).sort_values(by="rule")
    print("\n=== METRYKI (pozytyw = F3/F4, czyli FSCORE == '3-4') + dodatkowo metryki dla pozytywu = nie F3/F4 ===")
    print(report.to_string(index=False))

    # Zapis pełnego pliku z predykcjami
    df_out = df.copy()
    # Predykcje dla klasy "nie F3/F4" (negacja predykcji F3/F4)
    for _col in pred_cols + ["pred_ANY_OR"]:
        df_out[f"{_col}_notF3F4"] = ~df_out[_col].astype(bool)
    df_out["y_true_F3F4"] = y_true
    df_out["y_true_notF3F4"] = y_true_notF3F4
    df_out.to_csv(args.out, index=False)
    print(f"\nZapisano pełny plik z predykcjami: {args.out}")

    # Zapisy FP/FN dla głównej reguły ANY_OR
    os.makedirs(args.disagreements_dir, exist_ok=True)

    # Jeśli masz w danych identyfikator pacjenta/rekordu (np. ID), dopisz go do keep_cols.
    # Jeśli nie wiesz jakie są kolumny, zostaw None i dostaniesz cały rekord.
    keep_cols = None

    any_prefix = os.path.join(args.disagreements_dir, "ANY_OR")
    fp_path, fn_path = save_disagreements(df, y_true, "pred_ANY_OR", any_prefix, keep_cols=keep_cols)
    print(f"\nZapisano rozbieżności dla ANY_OR:")
    print(f"  FP: {fp_path}")
    print(f"  FN: {fn_path}")

    # (Opcjonalnie) FP/FN osobno dla każdej reguły
    if args.by_rule:
        per_dir = os.path.join(args.disagreements_dir, "disagreements_by_rule")
        os.makedirs(per_dir, exist_ok=True)
        for pred_name in pred_cols:
            rule = pred_name.replace("pred_", "")
            prefix = os.path.join(per_dir, rule)
            fp_p, fn_p = save_disagreements(df, y_true, pred_name, prefix, keep_cols=keep_cols)
            print(f"  {rule} -> FP: {fp_p} | FN: {fn_p}")


if __name__ == "__main__":
    main()
