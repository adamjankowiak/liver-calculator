import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data(path):
    return pd.read_csv(path)

def filter_age(df):
    return df[(df['Age']>35) & (df['Age']<65)]

def filter_hcv(df):
    return df[df['HCV'] == 1].copy()

def map_fib4_to_fscore_3_class(fib4, thresholds=(1.3, 2.67)):
    t0, t1= thresholds
    if fib4 < t0:
        return "1"
    elif fib4 < t1:
        return "2-3"
    else:
        return "4"

def map_fib4_to_fscore_binary(fib4, cutoff=1.3):
    return int(fib4 >= cutoff)

def evaluate_3_class_fib4(df, col_true='FSCORE', col_pred='f_pred'):
    y_true = df[col_true]
    y_pred = df[col_pred]
    print("=== 3-klasowa klasyfikacja===")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=["1","2-3","4"]))
    print(classification_report(y_true, y_pred,
                                labels=["1","2-3","4"],
                                target_names=['F1', 'F2-F3', 'F4']))
    print()

def evaluate_binary(df, col_true='y_true_bin', col_pred='y_pred_bin', col_score='FIB4'):
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
    filtered_hcv = filter_hcv(df).copy()
    filtered_age = filter_age(filtered_hcv).copy()
    #3 klasy - wiek 35-64
    filtered_age['f_pred'] = filtered_age['FIB4'].map(map_fib4_to_fscore_3_class)
    evaluate_3_class_fib4(filtered_age)

def run_binary(path_csv):
    df = load_data(path_csv)
    filtered_hcv = filter_hcv(df).copy()
    filtered_age = filter_age(filtered_hcv).copy()
    #2 klasy - wiek 35-64
    filtered_age['y_true_bin'] = filtered_age['FSCORE'].apply(lambda f: 0 if f == 1 else 1)
    filtered_age['y_pred_bin'] = filtered_age['FIB4'].map(map_fib4_to_fscore_binary)
    evaluate_binary(filtered_age)

if __name__ == "__main__":
    run_3_class("Data//FIB4//Fscore-No-Cat-Imp-3-Class-FIB4.csv")
    run_binary("Data//FIB4//Fscore-No-Cat-Imp-FIB4.csv")