import numpy as np
import pandas as pd


def bin_data(column):
    bins_scott = np.histogram_bin_edges(column, bins='scott')
    labels = [f"{bins_scott[i]}-{bins_scott[i+1]}" for i in range(len(bins_scott)-1)]
    return bins_scott, labels

def process_column(column):
    bins, labels = bin_data(column)
    column_bins = pd.cut(column, bins=bins, labels=labels, include_lowest=True)
    column_encoded = pd.get_dummies(column_bins, prefix=column.name)
    return column_encoded


df = pd.read_csv('homework5/heartRisk.csv')

df.columns = [col.lower() for col in df.columns]
df.columns = [col.lower() for col in df.columns]
columns_for_encoding = ['age', 'systolic', 'cholesterol', 'hdl']
for column in columns_for_encoding:
    df = pd.concat([df,  process_column(df[column])], axis=1)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype('int64')
