import pandas as pd
import numpy as np

import os

train_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\train.csv'
test_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\test.csv'

train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')

print(train_df.info())
test_df['Survived'] = -888 # Add Survived column for prediction
df = pd.concat((train_df, test_df), axis=0)
print("***INFO***")
print(df.info())
print("***HEAD***")
print(df.head())
print("***HEAD 10***")
print(df.head(10))
print("***TAIL***")
print(df.tail())
print("***DF NAME***")
print(df.Name)
# df['Name']
#print(df[['Name', 'Age']])
#print(df.loc[5:10,])
#print(df.loc[5:10, 'Age' : 'Pclass'])
print(df.loc[5:10, ['Survived', 'Fare','Embarked']])
male_passengers = df.loc[((df.Sex == 'male') & (df.Pclass == 1)),:]
print('Num of male passengers in first clas {0}'.format(len(male_passengers)))
