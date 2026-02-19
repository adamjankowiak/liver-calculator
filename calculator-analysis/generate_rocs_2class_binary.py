import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import apri_report
import pub_classification as pc


def roc_from_2class_preds_with_gray(
    df,
    index_name: str,
    truth_col: str = "FSCORE",
    pred_col: str = "f_pred",
    positive_label: str = "3-4",
    gray_label: str = "gray",
    out_dir: str = "roc_curves_2class_binary",
):
    """
    ROC/AUC liczone WYŁĄCZNIE z wyniku dwuklasowego (twarda predykcja):
      y_score = 1 jeśli pred == '3-4' else 0
      y_true  = 1 jeśli truth == '3-4' else 0

    Gray-zone w predykcjach jest usuwany.
    """
    if truth_col not in df.columns:
        raise KeyError(f"[{index_name}] Brak kolumny truth '{truth_col}' w danych.")
    if pred_col not in df.columns:
        raise KeyError(f"[{index_name}] Brak kolumny pred '{pred_col}' w danych.")

    sub = df[[truth_col, pred_col]].dropna().copy()

    # usuń gray-zone z predykcji
    sub = sub[sub[pred_col] != gray_label].copy()

    if sub.empty:
        print(f"[{index_name}] Brak danych po usunięciu gray-zone — ROC niepoliczalny.")
        return None

    y_true = (sub[truth_col].astype(str) == positive_label).astype(int).values
    y_score = (sub[pred_col].astype(str) == positive_label).astype(int).values

    # jeśli po filtracji w y_true jest tylko jedna klasa -> AUC nie istnieje
    if len(np.unique(y_true)) < 2:
        print(f"[{index_name}] Ground truth ma tylko jedną klasę po filtracji — ROC/AUC niepoliczalne.")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ROC_{index_name.replace(' ', '_').replace('-', '')}.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="losowy")
    plt.xlabel("False Positive Rate (1 - Swoistość)")
    plt.ylabel("True Positive Rate (Czułość)")
    plt.title(f"ROC (dwuklasowe predykcje) – {index_name} | Positive=F3-F4")
    plt.legend(loc="lower right")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[{index_name}] AUC = {auc_val:.4f} | zapisano: {out_path}")
    return auc_val


def run_and_plot_one(path_csv: str, score_col: str, map_func, index_name: str):
    df = apri_report.load_data(path_csv)

    # Twoja predykcja dwuklasowa z gray-zone
    df["f_pred"] = df[score_col].apply(map_func)

    # ROC tylko dla klasy 3-4 (positive)
    roc_from_2class_preds_with_gray(df, index_name=index_name)


if __name__ == "__main__":
    path = "../Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv"

    # APRI
    run_and_plot_one(
        path_csv=path,
        score_col="APRI",
        map_func=pc.map_apri_to_fscore_2_class_with_gray,
        index_name="APRI",
    )

    # FIB-4
    run_and_plot_one(
        path_csv=path,
        score_col="FIB4",
        map_func=pc.map_fib4_to_fscore_2_class_with_gray,
        index_name="FIB-4",
    )

    # NAFLD
    run_and_plot_one(
        path_csv=path,
        score_col="NAFLD",
        map_func=pc.map_nafld_to_fscore_2_class_with_gray,
        index_name="NAFLD",
    )

    # de Ritis
    run_and_plot_one(
        path_csv=path,
        score_col="RITIS",
        map_func=pc.map_ritis_to_fscore_2_class_with_gray,
        index_name="de_Ritis",
    )
