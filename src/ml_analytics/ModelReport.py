import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

import DataPreprocessing
import EDA
import ModelTrain


class ModelReport :
    df = input("Enter the dataframe: ")
    features = list(input("Enter the features: "))
    label = input("Enter the label: ")
    imputer = input("Enter the imputer handle missing values (KNNImputer, IterativeImputer, SimpleImputer, missForest): ")
    model_type = input("Enter the model type (classification, regression): ")
    test_size = float(input("Enter the test size (0.2, 0.3, 0.4): "))
    random_state = int(input("Enter the random state (0, 1, 2, 3, 4, 5): "))

    def __init__(self):
        self.df = ''
        self.features = ''
        self.label = ''
        self.imputer = ''
        self.model_type = ''
        self.test_size = 0.2
        self.random_state = 0

    #set values for the init method
    def set_values(self, df, features, label, imputer, model_type, test_size, random_state):
        self.df = df
        self.features = features
        self.label = label
        self.imputer = imputer
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state


    #fully automated report
    def report(self):
        """ fully automated report """
        #EDA
        eda = EDA(self.df)
        eda.describe()
        eda.separate()
        eda.hist()
        

        #DataPreprocessing
        data_preprocessing = DataPreprocessing(self.df)
        data_preprocessing.check_missing_value()
        data_preprocessing.duplicate_value()
        data_preprocessing.missing_value_handling(self.df, self.imputer)
        data_preprocessing.label_encoding(self.df, self.features)

        #ModelTrain
        model_train = ModelTrain(self.df, self.features, self.label, self.model_type, self.test_size, self.random_state)
        model_train.model_train()
