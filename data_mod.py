import numpy as np
import pandas as pd

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
df = pd.read_csv("Data/Modified/Fscore-No-Cat-Imp-2-Class.csv")
max_ast = 50
df['APRI'] = round((df['AST [U/l]'] / max_ast * 100) / df['PLT [k/µl]'],3)
df['FIB4'] = round((df['Age']*df['AST [U/l]'])/(df['PLT [k/µl]']*np.sqrt(df['ALT [U/l]'])),2)
df['RITIS'] = round(df['AST [U/l]']/df['ALT [U/l]'],2)
df['BMI'] = 20
df['NAFLD'] = round((0.037*df['Age']) +
                    (0.094*df["BMI"]) +
                    (1.13*df['Diabetes']) +
                    (0.99*df['AST [U/l]']/df['ALT [U/l]']) -
                    (0.013*df['PLT [k/µl]']) -
                    (0.66*df['Albumin [g/l]']*0.1) -
                    1.675, 3)

df.to_csv("Data/Full/Fscore-No-Cat-Imp-2-Class-Full.csv", index=False)