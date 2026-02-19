import pandas as pd

CALCULATORS = ["FIB4", "APRI", "RITIS", "NAFLD"]

def fscore_is_positive(series: pd.Series) -> pd.Series:
    """Definicja 'poprawnie przewidziane' = FSCORE == '3-4'."""
    return series.astype(str).str.strip().eq("3-4")

def alarms_has_calc(alarms_list: pd.Series, calc: str) -> pd.Series:
    """
    Sprawdza, czy dany kalkulator jest w alarms_list.
    alarms_list wygląda np. 'FIB4|APRI|NAFLD'.
    """
    s = alarms_list.fillna("").astype(str)
    # Bezpieczne dopasowanie tokenu w liście rozdzielanej '|'
    return s.str.contains(rf"(^|\|){calc}(\||$)", regex=True)

def report_by_calculators(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"pred_ANY_OR", "FSCORE", "alarms_list"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Brak wymaganych kolumn: {sorted(missing)}")

    subset = df[df["pred_ANY_OR"] == True].copy()
    total_or = len(subset)

    if total_or == 0:
        print("Brak rekordów z pred_ANY_OR == True (0 wierszy).")
        return pd.DataFrame()

    is_tp = fscore_is_positive(subset["FSCORE"])  # True Positive wg FSCORE

    # OVERALL (wszystkie OR-owe predykcje)
    overall_tp = int(is_tp.sum())
    overall_fp = int(total_or - overall_tp)

    rows = []
    for calc in CALCULATORS:
        mask_calc = alarms_has_calc(subset["alarms_list"], calc)
        y = int(mask_calc.sum())                    # wszystkie przewidzenia tego kalkulatora w OR==True
        x = int((mask_calc & is_tp).sum())          # poprawne przewidzenia (TP)
        fp = int(y - x)                             # false positive w ramach jego wskazań
        pct = (x / y * 100) if y else 0.0

        rows.append({
            "calculator": calc,
            "TP": x,
            "All predictions": y,
            "FP": fp,
            "TP rate": f"{pct:.2f}%",
        })

    report = pd.DataFrame(rows)

    # Wypisz czytelny raport tekstowy + zwróć DataFrame
    print("=== OVERALL (pred_ANY_OR == True) ===")
    print(f"TP: {overall_tp}/{total_or} ({overall_tp/total_or*100:.2f}%)")
    print(f"FP: {overall_fp}/{total_or} ({overall_fp/total_or*100:.2f}%)")
    print()
    print("=== BY CALCULATOR (udział w alarms_list przy OR==True) ===")
    print(report.to_string(index=False))

    return report

if __name__ == "__main__":
    report_by_calculators("Fscore-OR-predictions.csv")
