import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

train_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\train.csv'
test_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\test.csv'

train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')

print(train_df.info())
test_df['Survived'] = -888 #Add Survived column for prediction
df = pd.concat((train_df, test_df), axis=0)

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

