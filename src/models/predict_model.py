from src.data.data_extractor import data_extractor_df, data_extractor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pandas as pd
import pickle
from config import *

class PredictModel:

    def __init__(self):
        self.df = data_extractor_df
        self.train_df = data_extractor.get_processed_train_df()
        self.test_df = data_extractor.get_processed_test_df()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model_dummy = None
        self.model_lr_1 = None

    def prepare_data(self):
        X = self.train_df.loc[:, self.train_df.columns != 'Survived'].as_matrix().astype('float')
        y = self.train_df['Survived'].ravel()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # print(self.X_train.shape, self.y_train.shape)
        # print(self.X_test.shape, self.y_test.shape)

        # print("mean survival in train: {0:.3f}".format(np.mean(self.y_train)))
        # print("mean survival in test: {0:.3f}".format(np.mean(self.y_test)))

    def create_model(self):
        self.model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
        self.model_dummy.fit(self.X_train, self.y_train)
        print("score for baseline model: {0:.2f}".format(self.model_dummy.score(self.X_test, self.y_test)))

    def performance_metrics(self, model):
        accuracy_m = accuracy_score(self.y_test, model.predict(self.X_test))
        confusion_matrix_m = confusion_matrix(self.y_test, model.predict(self.X_test))
        precision_m = precision_score(self.y_test, model.predict(self.X_test))
        recall_m = self.y_test, model.predict(self.X_test)
        print("accuracy for model: {0:.2f}".format(accuracy_m))
        print("conf.matrix for model: \n {0}".format(confusion_matrix_m))
        print("precision for model: {0:.2f}".format(precision_m))
        print("recall for model: {0}".format(recall_m))

    def submission(self, model, file):
        test_X = self.test_df.as_matrix().astype('float')
        predictions = model.predict(test_X)
        df_submission = pd.DataFrame({'PassengerId': self.test_df.index, 'Survived': predictions})
        df_submission.head()
        df_submission.to_csv(file, index=False)

    def logistic_regression_model(self):
        self.model_lr_1 = LogisticRegression(random_state=0)
        self.model_lr_1.fit(self.X_train, self.y_train)
        print("score for logistic regression model: {0:.2f}".format(self.model_lr_1.score(self.X_test, self.y_test)))
        coef = self.model_lr_1.coef_
        print(coef)

    def hyperparameter_optimisation(self):
        model_lr = LogisticRegression(random_state=0)
        parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty':['l1', 'l2']}
        clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
        clf.fit(self.X_train, self.y_train)
        clf.best_params_
        print("best score {0:.2f}".format(clf.best_score_))
        print("logistic regression score {0:.2f}".format(clf.score(self.X_test, self.y_test)))
        self.submission(clf,SUBMISSION_FILE_PATH_LRM_OPTIMIZED)
        self.save_model_to_file(clf, LR_MODEL_FILE_PATH)

    def feature_normalization(self):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        print(X_train_scaled[:,0].min(), X_train_scaled[:,0].max())
        X_test_scaled = scaler.transform(self.X_test)

    def feature_standardization(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        print(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max())
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled

    def hyperparameter_optimisation_scaled(self):
        model_lr = LogisticRegression(random_state=0)
        parameters = {'C': [1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']}
        clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
        clf.fit(self.feature_standardization(), self.y_train)
        print("best score {0:.2f}".format(clf.best_score_))
        self.save_model_to_file(clf, LR_SCALED_MODEL_FILE_PATH)

    def save_model_to_file(self, model, file):
        model_file_pkl = open(file, 'wb')
        pickle.dump(model, model_file_pkl)
        model_file_pkl.close()

    def load_model_from_file(self, file):
        model_from_pkl = open(file, errors='ignore', mode='r')
        loaded = pickle.load(model_from_pkl)
        model_from_pkl.close()
        return loaded


pm = PredictModel()
pm.prepare_data()
pm.create_model()
# pm.logistic_regression_model()
# pm.performance_metrics(pm.model_lr_1)
# pm.submission(pm.model_lr_1, SUBMISSION_FILE_PATH_LRM)
#pm.hyperparameter_optimisation()
pm.feature_normalization()
pm.feature_standardization()
pm.hyperparameter_optimisation_scaled()
