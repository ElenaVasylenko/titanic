import matplotlib.pyplot as plt
from src.data.data_extractor import data_extractor_df

df = data_extractor_df

#df.plot.scatter(x='Age', y='Fare', color='c', title='scatter Age vs Fare', alpha=0.1)
df.plot.scatter(x='Pclass', y='Fare', color='c', title='scatter Pclass vs Pclass')
plt.show()