import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

train_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\train.csv'
test_file_path = 'D:\\UNI\\Uni IV\\titanic\\data_files\\test.csv'

train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')

print(train_df.info())
test_df['Survived'] = -888 # Add Survived column for prediction
df = pd.concat((train_df, test_df), axis=0)

#df.plot.scatter(x='Age', y='Fare', color='c', title='scatter Age vs Fare', alpha=0.1)
df.plot.scatter(x='Pclass', y='Fare', color='c', title='scatter Pclass vs Pclass')
plt.show()