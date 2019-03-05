import pandas as pd
import numpy as np
from src.data.data_extractor import data_extractor_df

df = data_extractor_df

gb = df.groupby('Sex').Age.median()
print(gb)
gb2 = df.groupby(['Pclass'])['Fare', 'Age'].median() # pclass ## fare | age
print(gb2)

gb3 = df.groupby(['Pclass']).agg({'Fare':'mean', 'Age':'median'})
print(gb3)

agr = {'Fare': {
    'mean_Fare':'mean',
    'median_Fare':'median',
    'max_Fare':max,
    'min_Fare':np.min
    },
    'Age': {
    'median_Age':'median',
    'min_Age': min,
    'max_Age': max,
    'range_Age': lambda x: max(x) - min(x)
    }
}

gb4 = df.groupby(['Pclass']).agg(agr)
gb5 = df.groupby(['Pclass', 'Embarked']).Fare.median()

print(gb4)
print(gb5)

