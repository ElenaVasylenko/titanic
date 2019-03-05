from src.data.data_extractor import data_extractor_df
import matplotlib.pyplot as plt

df = data_extractor_df

def descriptive_statistics(df):
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

    ## Univariate distribution

    # df.Pclass.value_counts().plot(kind='bar', rot=0,title='classwise passengers count', color='c')
    # plt.show()
    # df.Age.plot(kind='hist', title='histogram of passengers age',color='c', bins=20)
    #df.Age.plot(kind='kde', title='histogram of passengers age',color='c')
    df.Fare.plot(kind='hist', title='histogram for fare', color='c', bins=20)
    plt.show()
    print('skewness for age: {}'.format(df.Age.skew()))
    print('skewness for fare: {}'.format(df.Fare.skew()))

    ## Bivariate distribution

descriptive_statistics(df)