from titanic_data_processing.fix_missing_values import FixMissingValues
from src.data.data_extractor import data_extractor_df, data_extractor


class PredictModel:

    def __init__(self):
        self.df = data_extractor_df
        self.train_df = data_extractor.get_processed_train_df()
        self.test_df = data_extractor.get_processed_test_df()

    def prepare_data(self):
        print(self.train_df.info())
        print('-'*100)
        print(self.test_df.info())



pm = PredictModel()
pm.prepare_data()

