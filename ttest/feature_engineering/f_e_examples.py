from ttest.fix_missing_values import FixMissingValues
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mv = FixMissingValues()
df = mv.fix_nv_age(mv.df)

def age_state():
    df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')
    vc =df['AgeState'].value_counts()
    print("CROSSTAB ", vc)
    cs = pd.crosstab(df[df.Survived!= 888].Survived, df[df.Survived!= 888].AgeState)
    print(cs)
    #plt.show()

def family_size():
    df['FamilySize'] = df.Parch + df.SibSp + 1 # +1 for self
    df['FamilySize'].plot(kind='hist', color='g')
    plt.show()
    df.loc[df.FamilySize == df.FamilySize.max()] #explore families with max num of people
    cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != 888].FamilySize)
    print(cs)

def mother():
    df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch >0) & (df.Age >= 18) & (df.Title != 'Miss')), 1, 0)
    cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != 888].IsMother)
    print(cs)

def deck():
    print(df.Cabin)
    print(df.Cabin.unique())
    df.loc[df.Cabin == 'T', 'Cabin'] = np.NAN
    df['Deck'] = df['Cabin'].map(lambda x: get_deck(x))
    vc = df.Deck.value_counts()
    print(vc)
    cs = pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Deck)
    print(cs)

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')
#age_state()
#family_size()
#mother()
deck()