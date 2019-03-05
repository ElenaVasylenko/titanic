import pandas as pd
import numpy as np
from src.data.data_extractor import data_extractor_df
import matplotlib.pyplot as plt

class FixMissingValues:

    def __init__(self):
        self.df = data_extractor_df

    def fix_nv_embarked(self, df):
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

    def fix_nv_fare(self, df):
        print(df[df.Fare.isnull()])
        median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median()
        print(median_fare)
        df.Fare.fillna(median_fare, inplace=True)
        print(df.info())
        return df

    def get_title(self, name):
        title_group = {
            'mr': 'Mr',
            'miss': 'Miss',
            'ms': 'Miss',
            'master': 'Master',
            'don': 'Sir',
            'rev': 'Sir',
            'dr': 'Officer',
            'mme': 'Mrs',
            'mrs': 'Mrs',
            'major': 'Officer',
            'lady': 'Lady',
            'sir': 'Sir',
            'mlle': 'Miss',
            'col': 'Officer',
            'capt': 'Officer',
            'the countess': 'Lady',
            'jonkheer': 'Sir',
            'dona': 'Lady'
        }
        first_name_with_title = name.split(',')[1]
        title = first_name_with_title.split('.')[0]
        title = title.strip().lower()
        return title_group[title]

    def fix_nv_age(self, df):
        pd.options.display.max_rows = 15
        print(df[df.Age.isnull()])
        # df.Age.plot(kind='hist', bins=20)
        # plt.show()
        mean_age = df.Age.mean()
        print(mean_age)
        # df.Age.fillna(df.Age.mean(), inplase=True)

        df.groupby('Sex').Age.median()
        df[df.Age.notnull()].boxplot('Age', 'Sex')
        # df.Age.fillna(age_sex_median, inplase=True)

        df[df.Age.notnull()].boxplot('Age', 'Pclass')
        # pclass_age_median = df.groupby('Pclass').Age.transform('median')
        # df.Age.fillna(pclass_age_median, inplase=True)

        # 4-th way
        df.Name.map(lambda x: self.get_title(x))
        df.Name.map(lambda x: self.get_title(x)).unique()
        df['Title'] = df.Name.map(lambda x: self.get_title(x))
        df.head()
        df[df.Age.notnull()].boxplot('Age', 'Title')
        title_age_median = df.groupby('Title').Age.transform('median')
        df.Age.fillna(title_age_median, inplace=True)
        print(df.info())
        return df

    def get_df_wo_missing_values(self):
        self.fix_nv_embarked(df=self.df)
        self.fix_nv_fare(df=self.df)
        self.fix_nv_age(df=self.df)
        return self.df

    def treating_outliers(self, df):
        # df.Fare.plot(kind='hist', title='histogram for fare', bins=20, color='c')
        # plt.show()
        # log_fare = np.log(df.Fare + 1.0)
        pd.qcut(df.Fare, 4)
        pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])
        pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts()
        df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])

mv = FixMissingValues()
# mv.get_df_wo_missing_values()
mv.treating_outliers(mv.df)
mv.get_df_wo_missing_values()
CLEANED_DATA = mv.df
