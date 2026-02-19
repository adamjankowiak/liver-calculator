import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from mpmath.math2 import sqrt2

#CATEGORY
# df_cat = pd.read_csv("Data//Fscore-imputed-nan-del.csv")
# df_no_cat = pd.read_csv("Data//Fscore-No-Cat-Imp.csv")
#
# df_no_cat.insert(loc=24, column="Category", value=df_cat["Category"])
#
# df_no_cat.to_csv("Data//Fscore-Cat-Imp.csv")

#APRI
# df = pd.read_csv("Data//Fscore-No-Cat-Imp.csv")
# max_ast = 50
# df['APRI'] = round((df['AST [U/l]'] / max_ast * 100) / df['PLT [k/µl]'],3)
# df.to_csv("Data//Fscore-No-Cat-Imp.csv", index=False)

#FIB4
# df = pd.read_csv("Data//Fscore-No-Cat-Imp.csv")
# df['FIB4'] = round((df['Age']*df['AST [U/l]'])/(df['PLT [k/µl]']*np.sqrt(df['ALT [U/l]'])),2)
# df['FSCORE'] = df['FSCORE'].apply(lambda x: '2-3' if x in [2.0, 3.0] else ('1' if x == 1.0 else ('4' if x == 4.0 else x)))
# print(df.head(n=10))
# df.to_csv("Data//FIB4//Fscore-No-Cat-Imp-3-Class-FIB4.csv")

#RITIS
# df = pd.read_csv("Data//Modified//Fscore-No-Cat-Imp.csv")
# df['RITIS'] = round(df['AST [U/l]']/df['ALT [U/l]'],2)
# #df['FSCORE'] = df['FSCORE'].apply(lambda x: '2-3' if x in [2.0, 3.0] else ('1' if x == 1.0 else ('4' if x == 4.0 else x)))
# print(df.head(n=10))
# df.to_csv("Data//RITIS//Fscore-No-Cat-Imp-RITIS.csv")

# #NAFLD
# df1 = pd.read_csv("Data/NAFLD/Fscore-No-Cat-Imp-NAFLD.csv")
# df2 = pd.read_csv("Data/NAFLD/Fscore-No-Cat-Imp-3-Class-NAFLD.csv")
# df1['BMI'] = 30
# # df1['NAFLD'] = round((0.037*df1['Age']) +
# #                     (0.094*df1["BMI"]) +
# #                     (1.13*df1['Diabetes']) +
# #                     (0.99*df1['AST [U/l]']/df1['ALT [U/l]']) -
# #                     (0.013*df1['PLT [k/µl]']) -
# #                     (0.66*df1['Albumin [g/l]']*0.1) -
# #                     1.675, 3)
# df1.to_csv("Data//NAFLD//BMI_30//Fscore-No-Cat-Imp-NAFLD_BMI_30.csv")
# df2.to_csv("Data//NAFLD//BMI_30//Fscore-No-Cat-Imp-3-Class-NAFLD_BMI_30.csv")

#FULL DATA
# df = pd.read_csv("Data/Test-group/control.csv")
# exclude_cols = ["Virus","INR","Albumin [%]","PT [%]", "Gamma globulin [%]", "Gamma globulin [g/l]", "Bilirubin [µmol/l]", "Phosphatase alkaline [U/l]"]
# df.drop(exclude_cols, axis=1, inplace=True)
# impute_cols = (
#     df
#     .select_dtypes(include="number")
#     .columns
#     .difference(exclude_cols)
# )
#
# imputer = IterativeImputer(max_iter=50, min_value=0)
# df[impute_cols] = imputer.fit_transform(df[impute_cols])
# df = df.round(2)
#
#
# max_ast = 50
# df['Diabex'] = df['Diabex'].apply(lambda x:0 if x < 2 else 1)
# print(df['Diabex'])
# df['APRI'] = round((df['AST [U/l]'] / max_ast * 100) / df['PLT [k/µl]'],3)
# df['FIB4'] = round((df['Age']*df['AST [U/l]'])/(df['PLT [k/µl]']*np.sqrt(df['ALT [U/l]'])),2)
# df['RITIS'] = round(df['AST [U/l]']/df['ALT [U/l]'],2)
# df['BMI'] = 20
# df['NAFLD'] = round((0.037*df['Age']) +
#                     (0.094*df["BMI"]) +
#                     (1.13*df['Diabex']) +
#                     (0.99*df['AST [U/l]']/df['ALT [U/l]']) -
#                     (0.013*df['PLT [k/µl]']) -
#                     (0.66*df['Albumin [g/l]']*0.1) -
#                     1.675, 3)
#
# df.to_csv("Data/Test-group/control-imp.csv", index=False)
#
# df = pd.read_csv("Data/Test-group/control-imp.csv")
# df.describe()
# df.head()

# FSCORE 1,2,3,4 -> 1-2,3-4
df = pd.read_csv("Data/Modified/Full/Final_reduced/4-class/control.csv")
df['FSCORE'] = df['FSCORE'].apply(
    lambda x: '1-2' if x in [1.0, 2.0] else ('3-4' if x in [3.0, 4.0] else x))
df.to_csv("Data/Modified/Full/Final_reduced/2-class/control.csv", index=False)

#STANDARD SCALER
# from sklearn.preprocessing import StandardScaler
#
# df = pd.read_csv("Data/Modified/Full/Final_reduced/control.csv")
#
# exclude_cols = [
#     "Virus", "INR", "Albumin [%]", "PT [%]",
#     "Gamma globulin [%]", "Gamma globulin [g/l]",
#     "Bilirubin [µmol/l]", "Phosphatase alkaline [U/l]"
# ]
#
# df.drop(columns=exclude_cols, inplace=True, errors="ignore")
# never_scale = {"FSCORE", "FIBROSIS [kPa]", "STEATOSIS [dB/m]", "l.p.", "BMI"}
#
# binary_cols = [
#     c for c in df.columns
#     if c not in never_scale and df[c].dropna().nunique() == 2
# ]
#
# scale_cols = [
#     c for c in df.columns
#     if c not in never_scale
#     and c not in binary_cols
#     and pd.api.types.is_numeric_dtype(df[c])
# ]
#
# scaler = StandardScaler()
# df.loc[:, scale_cols] = scaler.fit_transform(df[scale_cols])
# df.to_csv("Data/Modified/Full/Final_reduced/Scaled/control.csv", index=False)
