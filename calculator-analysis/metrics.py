# metrics.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
)

def _print_per_class_metrics(cm, labels):
    """
    cm – macierz pomyłek (ndarray shape [n_classes, n_classes])
    labels – lista etykiet w tej samej kolejności co w cm
    """
    total = cm.sum()
    print("Per-class metrics:")
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FP - FN

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan          # precision
        npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan

        if (ppv + sensitivity) > 0:
            f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
        else:
            f1 = np.nan

        print(f"Klasa {label}:")
        print(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
        print(f"  Sensitivity (recall): {sensitivity:.3f}")
        print(f"  Specificity:          {specificity:.3f}")
        print(f"  PPV (precision):      {ppv:.3f}")
        print(f"  NPV:                  {npv:.3f}")
        print(f"  F1:                   {f1:.3f}")
        print()


def plot_confusion_matrix_like_example(
    cm,
    *,
    display_labels=("Zdrowy (1-2)", "Chory (3-4)"),
    title_prefix="FIB-4",
    n_eval=None,
    n_gray=None,
    save_path=None,
    show=True
):
    """
    Rysuje macierz pomyłek w stylu jak na załączonym obrazku:
    - osie: Rzeczywista klasa (Y), Predykcja (X)
    - etykiety: Zdrowy (1-2), Chory (3-4)
    - liczby w komórkach, pogrubione
    - tytuł z 'Do oceny (OUT+IN): ... | Wyłączono szarą strefę: ...'
    """
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    if n_eval is None:
        n_eval = int(cm.sum())
    if n_gray is None:
        n_gray = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # tick labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(display_labels, fontsize=14)
    ax.set_yticklabels(display_labels, fontsize=14)

    # osie
    ax.set_xlabel("Predykcja", fontsize=16)
    ax.set_ylabel("Rzeczywista klasa", fontsize=16)

    # tytuł
    ax.set_title(
        f"{title_prefix} – macierz pomyłek (bez szarej strefy)\n"
        f"Do oceny (OUT+IN): {n_eval} | Wyłączono szarą strefę: {n_gray}",
        fontsize=20,
        pad=18
    )

    # liczby w komórkach
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=18,
                color="white" if cm[i, j] > thresh else "black"
            )

    # delikatna siatka jak w przykładzie
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def evaluate_2_class_with_gray(
    df,
    col_true='FSCORE',
    col_pred='f_pred',
    gray_label='gray',
    *,
    plot_cm=True,
    cm_title_prefix="FIB-4",
    cm_save_path=None,
    show_plot=True
):
    """
    Wspólna ewaluacja 2-klasowa (F1-F2 vs F3-F4) z gray zone.

    Zakładamy, że:
      - kolumna z prawdą: col_true (domyślnie 'FSCORE') ma etykiety '1-2' / '3-4'
      - kolumna z predykcją: col_pred (domyślnie 'f_pred') ma etykiety '1-2' / '3-4' / gray_label
    """
    labels = ["1-2", "3-4"]
    target_names = ["F1-F2", "F3-F4"]

    n_total = len(df)
    mask_gray = (df[col_pred] == gray_label)
    n_gray = int(mask_gray.sum())
    n_eval = int(n_total - n_gray)

    print("=== 2-klasowa klasyfikacja (z gray zone) ===")
    print(f"Łączna liczba pacjentów:      {n_total}")
    print(f"Liczba w gray zone:           {n_gray} ({n_gray / n_total:.1%})")
    print(f"Liczba użyta do ewaluacji:    {n_eval} ({n_eval / n_total:.1%})")
    print()

    if n_eval == 0:
        print("Uwaga: wszystkie przypadki w gray zone – brak danych do policzenia metryk klasyfikacyjnych.")
        return

    df_eval = df[~mask_gray].copy()

    y_true = df_eval[col_true]
    y_pred = df_eval[col_pred]

    # macierz pomyłek (2x2)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # NOWE: rysowanie macierzy jak w przykładzie
    if plot_cm:
        plot_confusion_matrix_like_example(
            cm,
            display_labels=("Zdrowy (1-2)", "Chory (3-4)"),
            title_prefix=cm_title_prefix,
            n_eval=n_eval,
            n_gray=n_gray,
            save_path=cm_save_path,
            show=show_plot
        )

    # standardowy raport sklearn
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    # metryki ogólne
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    weighted_kappa = cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')

    print(f"Macro-F1:              {macro_f1:.3f}")
    print(f"Weighted-F1:           {weighted_f1:.3f}")
    print(f"Cohen's kappa:         {kappa:.3f}")
    print(f"Weighted kappa (quad): {weighted_kappa:.3f}")
    print()

    # per-klasa: TP, FP, TN, FN, sens, spec, PPV, NPV, F1
    _print_per_class_metrics(cm, labels)
