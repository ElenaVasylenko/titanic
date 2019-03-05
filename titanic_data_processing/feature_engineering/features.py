from titanic_data_processing.fix_missing_values import FixMissingValues, CLEANED_DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Features:

    def __init__(self):
        mv = FixMissingValues()
        self.df = CLEANED_DATA

    def age_state(self, df):
        df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')
        vc = df['AgeState'].value_counts()
        print("CROSSTAB ", vc)
        cs = pd.crosstab(df[df.Survived!= 888].Survived, df[df.Survived!= 888].AgeState)
        print(cs)
        #plt.show()

    def family_size(self, df):
        df['FamilySize'] = df.Parch + df.SibSp + 1 # +1 for self
        df['FamilySize'].plot(kind='hist', color='g')
        plt.show()
        df.loc[df.FamilySize == df.FamilySize.max()] #explore families with max num of people
        cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != 888].FamilySize)
        print(cs)

    def mother(self, df):
        df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch >0) & (df.Age >= 18) & (df.Title != 'Miss')), 1, 0)
        cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != 888].IsMother)
        print(cs)

    def get_deck(self, cabin):
        return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

    def deck(self, df):
        print(df.Cabin)
        print(df.Cabin.unique())
        df.loc[df.Cabin == 'T', 'Cabin'] = np.NAN
        df['Deck'] = df['Cabin'].map(lambda x: self.get_deck(x))
        vc = df.Deck.value_counts()
        print(vc)
        cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Deck)
        print(cs)

    def categorical_feature_encoding(self, df):
        df['IsMale'] = np.where(df.Sex == 'male', 1, 0)
        df = pd.get_dummies(df, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
        print('-'*60)
        print(df.info())
        return df

    def apply_all_features(self, df):
        self.age_state(df)
        self.family_size(df)
        self.mother(df)
        self.deck(df)

    def drop_unnecessary_columns(self, df):
        df.drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1, inplace=True)
        col = [column for column in df.columns if column != 'Survived']
        col = ['Survived']+ col
        df = df[col]
        print(10*'#')
        print(df.info())


features = Features()
features.apply_all_features(features.df)
features.df = features.categorical_feature_encoding(features.df)
features.drop_unnecessary_columns(features.df)
PROCESSED_DATA = features.df

# print(features.df.info())
#
#
# #age_state()
# #family_size()
# #mother()
# deck()