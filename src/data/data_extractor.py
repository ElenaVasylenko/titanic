import pandas as pd
from config import PROJECT_DIR_PATH, PROCESSED_FILES_DIR

class DataExtractor():

    def __init__(self):
        self.train_file_path = PROJECT_DIR_PATH + "\\data_files\\train.csv"
        self.test_file_path = PROJECT_DIR_PATH + "\\data_files\\test.csv"
        self.processed_train_file_path = PROCESSED_FILES_DIR + "\\train.csv"
        self.processed_test_file_path = PROCESSED_FILES_DIR + "\\test.csv"

        self.train_df = pd.read_csv(self.train_file_path, index_col='PassengerId')
        self.test_df = pd.read_csv(self.test_file_path, index_col='PassengerId')
        self.processed_train_df = pd.read_csv(self.processed_train_file_path, index_col='PassengerId')
        self.processed_test_df = pd.read_csv(self.processed_test_file_path, index_col='PassengerId')

        #print(self.train_df.info())
        self.test_df['Survived'] = -888  # Add Survived column for prediction
        self.df = pd.concat((self.train_df, self.test_df), axis=0)
        #print(self.df.info())

    def get_df(self):
        return self.df

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

    def get_processed_test_df(self):
        return self.processed_test_df

    def get_processed_train_df(self):
        return self.processed_train_df


data_extractor = DataExtractor()
data_extractor_df = data_extractor.get_df()
