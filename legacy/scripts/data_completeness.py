# missing_report.py
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

CSV_PATH = r"Data//Original//FibroScan-PUT-results.csv"  #


NA_VALUES = ["", " ","NaN", "nan", "NULL", "null", "-"]


def main():

    df = pd.read_csv(
        CSV_PATH,
        na_values=NA_VALUES,
        keep_default_na=True
    )

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
    df = df.replace({"": np.nan})

    candidates = [c for c in df.columns if str(c).lower() in {"id", "patient", "patient_id", "pacjent", "pacjent_id"}]
    if candidates:
        df = df.set_index(candidates[0])
        print(f"[INFO] Wykryto kolumnę identyfikatora pacjenta: '{candidates[0]}' – ustawiam jako indeks.")
    else:
        df.index = pd.Index([f"pacjent_{i+1}" for i in range(len(df))], name="pacjent")
        print("[INFO] Nie wykryto kolumny ID – tworzę sztuczny indeks 'row_#'.")

    # Kolumny „cech” (bez indeksu)
    feature_cols = list(df.columns)

    # === % braków per zmienna (kolumna) ===
    missing_per_col = df.isna().mean() * 100
    missing_per_col = missing_per_col.sort_values(ascending=False).round(2)
    missing_per_col_df = missing_per_col.reset_index()
    missing_per_col_df.columns = ["zmienna", "procent_brakow"]

    # === % braków per pacjent (wiersz) ===
    denom = max(len(feature_cols), 1)
    missing_per_row = df.isna().sum(axis=1) / denom * 100
    missing_per_row = missing_per_row.sort_values(ascending=False).round(2)
    missing_per_row_df = missing_per_row.reset_index()
    missing_per_row_df.columns = [df.index.name or "pacjent", "procent_brakow"]

    # === Zapisy wyników ===
    base_dir = os.path.dirname(os.path.abspath(CSV_PATH))
    out_cols_csv = os.path.join(base_dir, "missing_per_variable.csv")
    out_rows_csv = os.path.join(base_dir, "missing_per_patient.csv")
    missing_per_col_df.to_csv(out_cols_csv, index=False, encoding="utf-8")
    missing_per_row_df.to_csv(out_rows_csv, index=False, encoding="utf-8")

    # === Print: krótkie podsumowanie do konsoli ===
    pd.set_option("display.max_rows", 50)
    print("\n=== % braków per zmienna ===")
    print(missing_per_col_df.head(30).to_string(index=False))

    print("\n=== % braków per pacjent ===")
    print(missing_per_row_df.head(30).to_string(index=False))

    # === Heatmapa braków (pacjenci × zmienne) ===
    # Uwaga: przy bardzo dużych tabelach może to być ciężkie do odczytu.
    missing_matrix = df.isna().astype(int)  # 1 = brak, 0 = wartość obecna

    # Ograniczenie liczby pacjentów w heatmapie, jeśli danych jest bardzo dużo (opcjonalnie)
    MAX_ROWS_FOR_PLOT = 2000
    if len(missing_matrix) > MAX_ROWS_FOR_PLOT:
        print(f"[INFO] Dużo wierszy ({len(missing_matrix)}). Do heatmapy biorę pierwsze {MAX_ROWS_FOR_PLOT}.")
        missing_matrix_plot = missing_matrix.iloc[:MAX_ROWS_FOR_PLOT, :]
    else:
        missing_matrix_plot = missing_matrix

    plt.figure(figsize=(max(8, len(feature_cols) * 0.3), max(6, len(missing_matrix_plot) * 0.03)))

    sns.heatmap(
        missing_matrix_plot,
        cmap="viridis",  # 0–1 zróżnicowanie; 1 = brak
        cbar=True,
        yticklabels=False,
        xticklabels=True)


    plt.title("Heatmapa braków (1 = brak, 0 = wartość)")
    plt.xlabel("Zmienne")
    plt.ylabel("Pacjenci")
    plt.tight_layout()

    out_heatmap_png = os.path.join(base_dir, "missing_heatmap.png")
    plt.savefig(out_heatmap_png, dpi=200)
    print(f"\n[OK] Zapisano: {out_cols_csv}\n[OK] Zapisano: {out_rows_csv}\n[OK] Zapisano: {out_heatmap_png}")

    # Pokaż też na ekranie
    plt.show()


if __name__ == "__main__":
    main()
