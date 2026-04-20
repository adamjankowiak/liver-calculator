import pandas as pd

LOW_THRESHOLDS = {
    "FIB4": 1.45,        # < 1.45  => przewiduje 1-2
    "APRI": 0.5,         # < 0.5   => przewiduje 1-2
    "NAFLD": -1.455,     # < -1.455=> przewiduje 1-2
    "RITIS": 1.0,        # < 1.0   => przewiduje 1-2
}
CALCULATORS = list(LOW_THRESHOLDS.keys())

def fscore_is_low(series: pd.Series) -> pd.Series:
    """Ground truth: klasa 1-2."""
    return series.astype(str).str.strip().eq("1-2")

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def build_low_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje:
      - pred_<CALC>_LOW (True jeśli wynik < próg dolny)
      - pred_ANY_OR_LOW (OR po pred_*_LOW)
      - alarms_count_low, alarms_list_low
      - margin_low_<CALC> = value - threshold
      - y_true_1_2
    """
    out = df.copy()

    for calc, thr in LOW_THRESHOLDS.items():
        if calc not in out.columns:
            raise KeyError(f"Brak kolumny '{calc}' w danych.")
        x = to_num(out[calc])
        out[f"pred_{calc}_LOW"] = (x < thr).fillna(False)
        out[f"margin_low_{calc}"] = (x - thr).round(3)

    pred_cols = [f"pred_{c}_LOW" for c in CALCULATORS]
    out["pred_ANY_OR_LOW"] = out[pred_cols].any(axis=1)
    out["alarms_count_low"] = out[pred_cols].sum(axis=1).astype(int)

    def mk_list(row):
        hits = [c for c in CALCULATORS if row[f"pred_{c}_LOW"]]
        return "|".join(hits)

    out["alarms_list_low"] = out.apply(mk_list, axis=1)
    out["y_true_1_2"] = fscore_is_low(out["FSCORE"])
    return out

def alarms_has_calc_pipe(alarms_list: pd.Series, calc: str) -> pd.Series:
    s = alarms_list.fillna("").astype(str)
    return s.str.contains(rf"(^|\|){calc}(\||$)", regex=True)

def build_low_report_table(df_with_low: pd.DataFrame) -> pd.DataFrame:
    """
    Raport liczony wyłącznie na rekordach, gdzie pred_ANY_OR_LOW == True.
    TP = FSCORE == '1-2' w tej podgrupie.
    FP = pozostałe w tej podgrupie.
    """
    subset = df_with_low[df_with_low["pred_ANY_OR_LOW"] == True].copy()
    total_or = len(subset)
    if total_or == 0:
        return pd.DataFrame()

    is_tp = subset["y_true_1_2"] == True
    overall_tp = int(is_tp.sum())
    overall_fp = int(total_or - overall_tp)

    rows = [{
        "calculator": "OVERALL (ANY_OR_LOW)",
        "TP (X)": overall_tp,
        "All predictions (Y)": total_or,
        "FP (Y-X)": overall_fp,
        "TP rate (%)": round(overall_tp / total_or * 100, 2),
        "FP rate (%)": round(overall_fp / total_or * 100, 2),
    }]

    for calc in CALCULATORS:
        mask_calc = alarms_has_calc_pipe(subset["alarms_list_low"], calc)
        y = int(mask_calc.sum())
        x = int((mask_calc & is_tp).sum())
        fp = int(y - x)

        tp_rate = (x / y * 100) if y else 0.0
        fp_rate = (fp / y * 100) if y else 0.0

        rows.append({
            "calculator": calc,
            "TP (X)": x,
            "All predictions (Y)": y,
            "FP (Y-X)": fp,
            "TP rate (%)": round(tp_rate, 2),
            "FP rate (%)": round(fp_rate, 2),
        })

    return pd.DataFrame(rows)

def main():
    in_csv = "Fscore-OR-predictions.csv"

    df = pd.read_csv(in_csv)
    df_low = build_low_pred_columns(df)
    report = build_low_report_table(df_low)

    print(report.to_string(index=False))

    # Zapis tabeli raportowej
    report.to_csv("raport_kalkulatory_any_or_low.csv", index=False)
    report.to_excel("raport_kalkulatory_any_or_low.xlsx", index=False, sheet_name="report")

    # Opcjonalnie: zapis wzbogaconego zbioru z kolumnami LOW
    df_low.to_csv("Fscore-OR-predictions-with-LOW.csv", index=False)

if __name__ == "__main__":
    main()
