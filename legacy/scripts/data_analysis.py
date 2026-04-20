#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd

# === USTAWIENIA ===
INPUT = "Data/Original/FibroScan-PUT-results.csv"
SEP = ","            # jeśli masz średniki, ustaw na ";"
MAX_LABELS_PRINT = 2  # limit wypisywanych etykiet dla kategorycznych

# Popularne kody braku (możesz dopisać swoje)
MISSING = {
    "", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "?",
    "-", "--", -1, -9, -99, -999, -9999, 999, 9999, 99999
}

# --- POMOCNICZE REGEXY ---
# liczba z opcjonalnym znakiem i częścią ułamkową (kropka LUB przecinek)
_NUM_RE = re.compile(r"^\s*[+-]?\d+(?:[.,]\d+)?\s*$")

def parse_unit(col: str) -> str:
    m = re.search(r"\[([^\[\]]+)\]", str(col))
    return m.group(1).strip() if m else ""

def parse_desc(col: str) -> str:
    for sep in (" - ", " — "):
        if sep in str(col):
            return str(col).split(sep, 1)[1].strip()
    return ""

def detect_missing_codes(series: pd.Series) -> str:
    present = set()
    for v in series.dropna().unique().tolist():
        if isinstance(v, str):
            vv = v.strip()
            if vv in MISSING:
                present.add(vv)
        else:
            if v in MISSING:
                present.add(v)
    return ", ".join(map(str, sorted(present, key=lambda x: str(x)))) if present else ""

def coerce_missing_to_nan(series: pd.Series) -> pd.Series:
    s = series.copy()
    # ujednolicenie spacji
    if s.dtype == object:
        s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
        s = s.replace(list(MISSING), np.nan)
    else:
        num_tokens = {k: np.nan for k in MISSING if isinstance(k, (int, float))}
        s = s.replace(num_tokens)
    return s

def _to_numeric_with_decimal_comma(s: pd.Series) -> pd.Series:
    """
    Druga próba zrzutowania: jeżeli element wygląda jak liczba z przecinkiem,
    zamień przecinek dziesiętny na kropkę TYLKO dla takich elementów.
    """
    def _fix_one(x):
        if isinstance(x, str) and _NUM_RE.match(x):
            # jeśli ma przecinek w części dziesiętnej -> zamień na kropkę
            # (nie ruszamy separatorów tysięcy, bo ich tu nie zakładamy)
            if "," in x and "." not in x:
                return x.replace(",", ".")
        return x

    fixed = s.map(_fix_one)
    return pd.to_numeric(fixed, errors="coerce")

def looks_numeric(series: pd.Series):
    """
    Próbuje zrzutować na liczby. Uznajemy kolumnę za numeryczną, jeśli
    >= 90% niepustych wartości daje się zamienić na liczbę.
    Zwraca (is_numeric, numeric_series).
    """
    s = coerce_missing_to_nan(series)

    # 1. standardowa próba
    numeric1 = pd.to_numeric(s, errors="coerce")
    non_na = s.notna()
    if non_na.sum() == 0:
        return False, numeric1
    share1 = numeric1.notna().sum() / non_na.sum()
    if share1 >= 0.9:
        return True, numeric1

    # 2. próba z przecinkiem dziesiętnym
    numeric2 = _to_numeric_with_decimal_comma(s)
    share2 = numeric2.notna().sum() / non_na.sum()
    if share2 >= 0.9 or share2 > share1:
        return share2 >= 0.9, numeric2

    return False, numeric1  # domyślnie nie-numeryczna

def fmt(x, nd=4):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    return f"{float(x):.{nd}f}"

def main():
    df = pd.read_csv(INPUT, sep=SEP)

    # pomiń ostatnie dwie kolumny
    if df.shape[1] >= 2:
        df = df.iloc[:, :-2].copy()

    for col in df.columns:
        raw = df[col]
        unit = parse_unit(col)
        desc = parse_desc(col)
        miss_codes = detect_missing_codes(raw)

        is_num, num_series = looks_numeric(raw)
        print(f"Nazwa: {col}")
        if desc:
            print(f"Opis kliniczny: {desc}")
        if unit:
            print(f"Jednostka: {unit}")
        if miss_codes:
            print(f"Kod braku: {miss_codes}")

        if is_num:
            s = num_series.dropna()
            if len(s) == 0:
                print("Zakres: ")
                print("Średnia: ")
                print("Mediana: ")
                print("Odchylenie standardowe (ddof=1): ")
            else:
                print(f"Zakres: [{fmt(s.min())}, {fmt(s.max())}]")
                print(f"Średnia: {fmt(s.mean())}")
                print(f"Mediana: {fmt(s.median())}")
                std_val = s.std(ddof=1) if len(s) > 1 else 0.0  # próbka (n-1)
                print(f"Odchylenie standardowe (ddof=1): {fmt(std_val)}")
        else:
            # Etykiety tylko dla nienumerycznych
            s = coerce_missing_to_nan(raw).dropna().astype("object")
            s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
            labels = sorted(map(str, s.unique()))
            total = len(labels)
            if total == 0:
                print("Wszystkie etykiety: (brak)")
            else:
                shown = labels[:MAX_LABELS_PRINT]
                print(f"Wszystkie etykiety ({min(total, MAX_LABELS_PRINT)}/{total}):")
                for lab in shown:
                    print(f" - {lab}")
                if total > MAX_LABELS_PRINT:
                    print(f" ... (+{total - MAX_LABELS_PRINT} więcej)")
        print("")

if __name__ == "__main__":
    main()
