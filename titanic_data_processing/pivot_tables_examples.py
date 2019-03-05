import pandas as pd
from src.data.data_extractor import data_extractor_df

df = data_extractor_df

def pivot_example(df):
    pv = df.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean')
    gb1 = df.groupby(['Sex', 'Pclass']).Age.mean()
    gb2 = df.groupby(['Sex', 'Pclass']).Age.mean().unstack()

    # print('-'*10)
    # print(gb1)
    # print('-'*10)
    # print(gb2)
    # print('-'*10)
    # print(pv)
pivot_example(df)
