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

print(df.info())

def fix_nv_embarked(df):
    null_em = df[df.Embarked.isnull()]
    print(null_em)
    v_c_em = df.Embarked.value_counts()
    pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Embarked)
    # df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'
    # df.Embarked.fillna('S', inplace=True)
    df.groupby(['Pclass', 'Embarked']).Fare.median()
    df.Embarked.fillna('C', inplace=True)
    null_em1 = df[df.Embarked.isnull()]
    print(df.info())
    return df

def fix_nv_fare(df):
    print(df[df.Fare.isnull()])
    median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median()
    print(median_fare)
    df.Fare.fillna(median_fare, inplace=True)
    print(df.info())
    return df

def fix_nv_age(df):
    pd.options.display.max_rows = 15
    print(df[df.Age.isnull()])
    df.Age.plot(kind='hist', bins=20)
    # plt.show()
    mean_age = df.Age.mean()
    print(mean_age)
    #df.Age.fillna(df.Age.mean(), inplase=True)

    df.groupby('Sex').Age.median()
    df[df.Age.notnull()].boxplot('Age', 'Sex')
    #df.Age.fillna(age_sex_median, inplase=True)

    df[df.Age.notnull()].boxplot('Age', 'Pclass')
    #pclass_age_median = df.groupby('Pclass').Age.transform('median')
    #df.Age.fillna(pclass_age_median, inplase=True)

    # 4-th way
    df.Name.map(lambda x:get_title(x))
    df.Name.map(lambda x:get_title(x)).unique()
    df['Title'] = df.Name.map(lambda x:get_title(x))
    df.head()
    df[df.Age.notnull()].boxplot('Age', 'Title')
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    print(df.info())
    return df

def get_title(name):
    title_group = {
        'mr':'Mr',
        'miss':'Miss',
        'ms':'Miss',
        'master':'Master',
        'don':'Sir',
        'rev':'Sir',
        'dr':'Officer',
        'mme':'Mrs',
        'mrs':'Mrs',
        'major':'Officer',
        'lady':'Lady',
        'sir':'Sir',
        'mlle':'Miss',
        'col':'Officer',
        'capt':'Officer',
        'the countess':'Lady',
        'jonkheer':'Sir',
        'dona':'Lady'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]



