import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class DataPreprocessing :

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    def __init__(self, df):
        self.df = df

    #Check missing value 
    def check_missing_value(self,df):
        """ Check missing value """

        def __str__(self):
            return "Missing value"

        check_missing = (df.isnull().sum() * 100 / df.shape[0]).sort_values(ascending=False) 
        check_missing[check_missing > 0]
        print(check_missing)
        return check_missing

    #Check duplicate value
    def duplicate_value(self,df):

        """ Check duplicate value """
        check_duplicate = df.duplicated().sum()
        print(check_duplicate)
        return check_duplicate

    #Handling missing values with KNNImputer, IterativeImputer, SimpleImputer, missForest for particular column
    def missing_value_handling(self, column, method):
        """ Handling missing values with KNNImputer, IterativeImputer, SimpleImputer, missForest for particular column """
        if method == 'KNNImputer':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df[column] = imputer.fit_transform(df[[column]])
        elif method == 'IterativeImputer':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=0)
            df[column] = imputer.fit_transform(df[[column]])
        elif method == 'SimpleImputer':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            df[column] = imputer.fit_transform(df[[column]])
        elif method == 'missForest':
            from missingpy import MissForest
            imputer = MissForest()
            df[column] = imputer.fit_transform(df[[column]])
        else:
            print('Please choose the method')

    ##Handling missing value for all columns automatically with KNNImputer only
    def missing_value_handling_all(self,df):
        """ Handling missing value for all columns automatically with KNNImputer only """
        if method == 'KNNImputer':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df = imputer.fit_transform(df)
        else:
            print('Please choose the method')


            
    #Label Encoding for Object datatype all columns automatically
    def label_encoding_all(self,df):
        """ Label Encoding for Object datatype all columns automatically """
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])
        return df


    #One Hot Encoding for Object datatype column
    def one_hot_encoding(self, df):
        """ One Hot Encoding for Object datatype column """
        # Create dummy variables using pandas
        for col in df.column:
            if df[col].dtype == 'object':
                df = pd.get_dummies(df, columns=[col], prefix=[col])

    #Perform one hot encoding for multicategorical column select top 10
    def one_hot_encoding_top10(self, column):
        """ Perform one hot encoding for multicategorical column select top 10 """
        # Create dummy variables using pandas
        top_10 = [x for x in df[column].value_counts().sort_values(ascending=False).head(10).index]
        for label in top_10:
            df[column+'_'+label] = np.where(df[column]==label, 1, 0)
        return df
        
    #Correlation Heatmap for all columns
    def correlation_heatmap(self,df):
        """ Correlation Heatmap for all columns """
        corr = df.corr()
        plt.figure(figsize=(18, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
        plt.show()
        
    #Remove column with correlation value more than 0.9
    def remove_column_with_high_corr(self,df):
        """ Remove column with correlation value more than 0.9 """
        corr = df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        df.drop(to_drop, axis=1, inplace=True)
        print("Columns dropped: ", to_drop)
        
    
    