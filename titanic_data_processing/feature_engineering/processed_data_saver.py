from titanic_data_processing.feature_engineering.features import PROCESSED_DATA
from config import PROJECT_DIR_PATH
import os

df = PROCESSED_DATA
print(df.info())
processed_data_path = os.path.join(PROJECT_DIR_PATH,'data_files', 'processed')
train_path = os.path.join(processed_data_path, 'train.csv')
test_path = os.path.join(processed_data_path, 'test.csv')

#train
df.loc[df.Survived != -888].to_csv(train_path)
#test
columns = [col for col in df.columns if col != 'Survived']
df.loc[df.Survived != -888, columns].to_csv(test_path)