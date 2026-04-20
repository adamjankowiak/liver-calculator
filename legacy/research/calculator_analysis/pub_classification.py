import apri_report
from apri_3_class_report import map_apri_to_fscore_3_class
from fib4_report import map_fib4_to_fscore_3_class
from nafld_report import map_nafld_to_fscore_3_class
from metrics import evaluate_2_class_with_gray

def _map_3class_label_to_2class_with_gray(label_3: str) -> str:
    """
    Przyjmuje etykietę 3-klasową:
        '1', '2-3', '4'
    i zwraca etykietę 2-klasową z gray zone:
        '1-2', '3-4' lub 'gray'
    """
    if label_3 == "1":
        return "1-2"
    elif label_3 == "4":
        return "3-4"
    else:
        # '2-3' (lub cokolwiek innego) traktujemy jako gray zone
        return "gray"

def map_apri_to_fscore_2_class_with_gray(apri: float) -> str:
    """
    APRI:
      gray zone: 0.5 – 1.0
      F1–F2: APRI < 0.5
      F3–F4: APRI > 1.0
    """
    # thresholds=(0.5, 1.0) jest domyślne, ale podajemy jawnie
    label_3 = map_apri_to_fscore_3_class(apri, thresholds=(0.5, 1.5))
    return _map_3class_label_to_2class_with_gray(label_3)


def map_fib4_to_fscore_2_class_with_gray(fib4: float) -> str:
    """
    FIB-4:
      gray zone: 1.45 – 3.25
      F1–F2: FIB-4 < 1.45
      F3–F4: FIB-4 > 3.25
    """
    label_3 = map_fib4_to_fscore_3_class(fib4, thresholds=(1.45, 3.25))
    return _map_3class_label_to_2class_with_gray(label_3)


def map_nafld_to_fscore_2_class_with_gray(nafld: float) -> str:
    """
    NAFLD:
      gray zone: -1.455 – 0.676
      F1–F2: NAFLD < -1.455
      F3–F4: NAFLD > 0.676
    """
    label_3 = map_nafld_to_fscore_3_class(nafld, thresholds=(-1.455, 0.676))
    return _map_3class_label_to_2class_with_gray(label_3)


def map_ritis_to_fscore_2_class_with_gray(ritis: float, cutoff: float = 1.5) -> str:
    """
    de Ritis:
      gray zone NIE istnieje.
      F1–F2: de Ritis <= 1
      F3–F4: de Ritis > 1

    Funkcja zwraca tylko '1-2' lub '3-4' (bez 'gray'),
    więc w ewaluacji n_gray = 0.
    """
    if ritis <= cutoff:
        return "1-2"
    else:
        return "3-4"


def _run_index_2_class(path_csv: str, score_col: str, map_func, index_name: str):
    """
    Wspólna funkcja uruchamiająca 2-klasową klasyfikację
    (F1-F2 vs F3-F4) z gray zone (jeśli map_func generuje 'gray')
    dla wybranego kalkulatora.
    """
    df = apri_report.load_data(path_csv)

    # wszyscy pacjenci
    df['f_pred'] = df[score_col].apply(map_func)
    print(f"=== {index_name}: wszystkie przypadki ===")
    evaluate_2_class_with_gray(df)

    # # tylko HCV+
    # df_hcv = apri_report.filter_hcv(df)
    # df_hcv['f_pred'] = df_hcv[score_col].apply(map_func)
    # print(f"=== {index_name}: tylko HCV+ ===")
    # evaluate_2_class_with_gray(df_hcv)


def run_2_class_APRI(path_csv: str):
    _run_index_2_class(
        path_csv=path_csv,
        score_col='APRI',
        map_func=map_apri_to_fscore_2_class_with_gray,
        index_name='APRI'
    )


def run_2_class_FIB4(path_csv: str):
    _run_index_2_class(
        path_csv=path_csv,
        score_col='FIB4',
        map_func=map_fib4_to_fscore_2_class_with_gray,
        index_name='FIB-4'
    )


def run_2_class_NAFLD(path_csv: str):
    _run_index_2_class(
        path_csv=path_csv,
        score_col='NAFLD',
        map_func=map_nafld_to_fscore_2_class_with_gray,
        index_name='NAFLD'
    )


def run_2_class_RITIS(path_csv: str):
    _run_index_2_class(
        path_csv=path_csv,
        score_col='RITIS',
        map_func=map_ritis_to_fscore_2_class_with_gray,
        index_name='de Ritis'
    )


if __name__ == '__main__':
    path = "../Data/Modified/Full/Final_reduced/2-class/control.csv"

    # APRI
    run_2_class_APRI(path)

    # FIB-4
    run_2_class_FIB4(path)

    # NAFLD
    run_2_class_NAFLD(path)

    # de Ritis
    run_2_class_RITIS(path)
