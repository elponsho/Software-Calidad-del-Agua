import pandas as pd

def correlacion_pearson(df):
    return df.corr(method='pearson')

def correlacion_spearman(df):
    return df.corr(method='spearman')
