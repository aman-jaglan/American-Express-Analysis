import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os
pd.set_option('display.max_columns', None)


#%%

import dask.dataframe as dd
#
df = dd.read_csv('train_data.csv')
print(df.head())

df = df.compute()

# Step 2: List of specific columns you want to keep
columns_to_keep = [
    'customer_ID', 'S_2', 'S_3', 'P_2', 'D_39', 'B_1', 'B_2',
    'R_1', 'B_30', 'D_114', 'D_116'
]

# Step 3: Select the specific columns from the DataFrame
df_selected = df[columns_to_keep]

# Step 4: Select the first 128298 rows
df_selected = df_selected.iloc[:128298]

df_selected.to_csv('data.csv', index=False)

