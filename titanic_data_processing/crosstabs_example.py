import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_extractor import data_extractor_df

df = data_extractor_df

def crosstab_example_Sex_Pclass(df):
    cs = pd.crosstab(df.Sex, df.Pclass)
    print(cs)
    pd.crosstab(df.Sex, df.Pclass).plot(kind='bar')
    plt.show()

crosstab_example_Sex_Pclass(df)