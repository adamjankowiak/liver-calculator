import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data(path):
    return pd.read_csv(path)

def map_ritis_to_fscore_3_class(ritis, thresholds=(1.5, 2.0)):
    t0, t1= thresholds
    if ritis < t0:
        return "1"
    elif ritis < t1:
        return "2-3"
    else:
        return "4"

def map_ritis_to_fscore_binary(ritis, cutoff=1.5):
    return int(ritis >= cutoff)

def evaluate_3_class_ritis(df, col_true='FSCORE', col_pred='f_pred'):
    y_true = df[col_true]
    y_pred = df[col_pred]
    print("=== 3-klasowa klasyfikacja===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=["1","2-3","4"]))
    print(classification_report(y_true, y_pred,
                                labels=["1","2-3","4"],
                                target_names=['F1', 'F2-F3', 'F4']))
    print()

def evaluate_binary_ritis(df, col_true='y_true_bin', col_pred='y_pred_bin', col_score='RITIS'):
    y_true = df[col_true]
    y_pred = df[col_pred]
    y_score = df[col_score]
    print("=== Binarna klasyfikacja===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred,
                                target_names=['F1', 'F2-F4']))
    print(f"AUC: {roc_auc_score(y_true, y_score):.3f}")

def run_3_class(path_csv):
    df = load_data(path_csv)
    df['f_pred'] = df['RITIS'].apply(map_ritis_to_fscore_3_class)
    evaluate_3_class_ritis(df)

def run_binary(path_csv):
    df = load_data(path_csv)
    df['y_true_bin'] = df['FSCORE'].apply(lambda f: 0 if f == 1 else 1)
    df['y_pred_bin'] = df['RITIS'].map(map_ritis_to_fscore_binary)
    evaluate_binary_ritis(df)

if __name__ == '__main__':
    run_3_class("Data/RITIS/Fscore-No-Cat-Imp-3-Class-RITIS.csv")
    run_binary("Data/RITIS/Fscore-No-Cat-Imp-RITIS.csv")
