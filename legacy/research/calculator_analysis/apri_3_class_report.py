import numpy as np
import apri_report
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
)

def map_apri_to_fscore_3_class(apri, thresholds=(0.5, 1.0)):
    t0, t1 = thresholds
    if apri < t0:
        return "1"
    elif apri < t1:
        return "2-3"
    else:
        return "4"


def _print_per_class_metrics(cm, labels):
    """
    cm – macierz pomyłek (ndarray shape [n_classes, n_classes])
    labels – lista etykiet w tej samej kolejności co w cm
    """
    total = cm.sum()
    print("Per-class metrics (one-vs-rest):")
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


def evaluate_3_class(df, col_true='FSCORE', col_pred='f_pred'):
    labels = ["1", "2-3", "4"]
    target_names = ["F1", "F2-F3", "F4"]

    y_true = df[col_true]
    y_pred = df[col_pred]

    print("=== 3-klasowa klasyfikacja ===")

    # macierz pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix:\n", cm)
    print()

    # standardowy raport sklearn (zawiera m.in. per-class precision, recall, F1)
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names
    ))

    # metryki wieloklasowe
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


def run_3_class_APRI(path_csv):
    df = apri_report.load_data(path_csv)

    # 3 klasy - wszystkie rekordy
    df['f_pred'] = df['APRI'].apply(map_apri_to_fscore_3_class)
    evaluate_3_class(df)

    # 3 klasy - tylko HCV+
    df_hcv = apri_report.filter_hcv(df)
    df_hcv['f_pred'] = df_hcv['APRI'].apply(map_apri_to_fscore_3_class)
    evaluate_3_class(df_hcv)


if __name__ == '__main__':
    run_3_class_APRI("../Data/NAFLD/Fscore-No-Cat-Imp-3-Class-NAFLD.csv")
