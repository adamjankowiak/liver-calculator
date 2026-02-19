import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



def binary_metrics_from_cm(y_true, y_pred, pos_label="3-4"):
    """
    Metryki dla problemu binarnego na bazie macierzy pomyłek:
    - recall / sensitivity (TPR)
    - specificity (TNR)
    - PPV (precision)
    - NPV
    Działa dla etykiet stringowych typu "1-2" / "3-4".
    """
    classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    if len(classes) != 2:
        raise ValueError(f"To nie jest problem binarny: classes={classes}")
    if pos_label not in classes:
        raise ValueError(f"pos_label={pos_label} nie występuje w danych: classes={classes}")

    neg_label = classes[classes != pos_label][0]
    labels = [neg_label, pos_label]  # [neg, pos]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0                 # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0            # TNR
    ppv = tp / (tp + fp) if (tp + fp) else 0.0                    # precision
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "pos_label": pos_label,
        "neg_label": neg_label,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": recall,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }


param_grid = {
    'max_depth': [2, 3],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'ccp_alpha': [0.0, 0.01],
}

train_df = pd.read_csv("Data/Modified/Full/Final/Reduced-Imp-Iterative-2-Class.csv")
test_df  = pd.read_csv("Data/Test-group/control-imp-2-class.csv")

target_col = "FSCORE"
feature_cols = ["FIB4", "RITIS", "APRI", "NAFLD"]
#feature_cols = [ "Age", "PLT [k/µl]", "AST [U/l]", "ALT [U/l]", "Albumin [g/l]", "FIB4", "APRI", "RITIS", "NAFLD"]

train_df[target_col] = train_df[target_col].astype(str).str.strip()
test_df[target_col]  = test_df[target_col].astype(str).str.strip()

allowed = {"1-2", "3-4"}
train_df = train_df[train_df[target_col].isin(allowed)].copy()
test_df  = test_df[test_df[target_col].isin(allowed)].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test  = test_df[feature_cols].copy()
y_test  = test_df[target_col].copy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_model = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid.fit(X_train, y_train)

print("Najlepsze parametry:", grid.best_params_)
print("Najlepszy wynik CV (średni):", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


m = binary_metrics_from_cm(y_test, y_pred, pos_label="3-4")
print("\nDodatkowe metryki (pozytywna klasa = '3-4'):")
print(f"  Recall / Sensitivity (TPR): {m['recall']:.4f}")
print(f"  Specificity (TNR):          {m['specificity']:.4f}")
print(f"  PPV (Precision):            {m['ppv']:.4f}")
print(f"  NPV:                        {m['npv']:.4f}")
print(f"  Counts (tn, fp, fn, tp):    {m['tn']}, {m['fp']}, {m['fn']}, {m['tp']}")


plt.figure(figsize=(18, 9))
plot_tree(
    best_model,
    feature_names=X_train.columns,
    class_names=[str(c) for c in best_model.classes_],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    fontsize=14
)
plt.tight_layout()
plt.savefig("plot_control_model_full_features.png")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# prawdziwe etykiety w danych:
labels_true = ["1-2", "3-4"]

# ładne napisy do osi (kolejność musi odpowiadać labels_true):
labels_pretty = ["F1-F2", "F3-F4"]

cm = confusion_matrix(y_test, y_pred, labels=labels_true)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_pretty)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(values_format="d", cmap="Blues", colorbar=False, ax=ax)

# <- tutaj zmieniasz “True label” i “Predicted label” na polski:
ax.set_xlabel("Klasa przewidziana")   # zamiast Predicted label
ax.set_ylabel("Klasa rzeczywista")    # zamiast True label

ax.set_title("Macierz pomyłek - Drzewo decyzyjne, FSCORE: 1-2 vs. 3-4")
fig.tight_layout()
fig.savefig("confusion_matrix_tree_4_att.png", dpi=300)
plt.show()

