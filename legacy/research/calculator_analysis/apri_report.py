import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    roc_auc_score
)
from metrics import _print_per_class_metrics

def load_data(path):
    return pd.read_csv(path)

def filter_hcv(df):
    return df[df['HCV'] == 1].copy()

def map_apri_to_fscore_multi(apri, thresholds=(0.5, 0.7, 1.0)):

    t0, t1, t2 = thresholds
    if apri < t0:
        return 1
    elif apri < t1:
        return 2
    elif apri < t2:
        return 3
    else:
        return 4

def map_apri_to_binary(apri, cutoff=0.7):
    return int(apri >= cutoff)

def evaluate_multi(df, col_true='FSCORE', col_pred='f_pred'):
    #Metryki dla 4 klascha
    y_true = df[col_true]
    y_pred = df[col_pred]
    print("=== 4-klasowa klasyfikacja ===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[1,2,3,4]))
    print(classification_report(y_true, y_pred,
                                labels=[1,2,3,4],
                                target_names=[
                                    "F1",
                                    "F2",
                                    "F3",
                                    "F4"]))
    print()

    _print_per_class_metrics(cm, labels)

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    weighted_kappa = cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')

    print(f"Macro-F1:              {macro_f1:.3f}")
    print(f"Weighted-F1:           {weighted_f1:.3f}")
    print(f"Cohen's kappa:         {kappa:.3f}")
    print(f"Weighted kappa (quad): {weighted_kappa:.3f}")
    print()


def evaluate_binary(df, col_true='y_true_bin', col_pred='y_pred_bin', col_score='APRI'):
    #Metryki dla 2 klas
    y_true = df[col_true]
    y_pred = df[col_pred]
    y_score = df[col_score]
    print("=== Binarna klasyfikacja ===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['F1–2 (low)', 'F3–4 (high)']))
    print(f"AUC: {roc_auc_score(y_true, y_score):.3f}")
    print()

def run_all(path_csv):
    df = load_data(path_csv)

    #4 klasy – wszystkie rekordy
    df['f_pred'] = df['APRI'].apply(map_apri_to_fscore_multi)
    evaluate_multi(df)

    #2 klasy – wszystkie rekordy
    df['y_true_bin'] = df['FSCORE'].apply(lambda f: 0 if f <= 2 else 1)
    df['y_pred_bin'] = df['APRI'].apply(map_apri_to_binary)
    evaluate_binary(df)

    print("=== HCV ===")

    #4 klasy – tylko HCV+
    df_hcv = filter_hcv(df)
    df_hcv['f_pred'] = df_hcv['APRI'].apply(map_apri_to_fscore_multi)
    evaluate_multi(df_hcv)

    # 2 klasy – tylko HCV+
    df_hcv['y_true_bin'] = df_hcv['FSCORE'].apply(lambda f: 0 if f <= 2 else 1)
    df_hcv['y_pred_bin'] = df_hcv['APRI'].apply(map_apri_to_binary)
    evaluate_binary(df_hcv)

if __name__ == "__main__":
    run_all("../Data/Modified/Fscore-No-Cat-Imp.csv")