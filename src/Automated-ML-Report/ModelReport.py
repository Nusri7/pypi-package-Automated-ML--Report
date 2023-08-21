import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class ModelTrain:
    def __init__(self, df, x, y, test_size):
        self.df = df
        self.x = x
        self.y = y
        self.test_size = test_size
        
        

    #Splitting the dataset into the Training set and Test set
    def split(self,df):
        

        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        return X_train, X_test, y_train, y_test

    #Remove the Outlier from train data using ZScore method
    def outlier_zscore(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from scipy import stats
        z = np.abs(stats.zscore(X_train))
        print(z)
        threshold = 3
        print(np.where(z > 3))
        X_train = X_train[(z < 3).all(axis=1)]
        return X_train

    #Remove the Outlier from train data using IQR method
    def outlier_iqr(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        Q1 = X_train.quantile(0.25)
        Q3 = X_train.quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)
        print((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 * IQR)))
        X_train = X_train[~((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 * IQR))).any(axis=1)]
        return X_train

    #Remove the Outlier from train data using Isolation Forest method
    def outlier_isolation(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(random_state=0).fit(X_train)
        X_train = X_train[clf.predict(X_train) == 1]
        return X_train

    #Perform the feature scaling on train data
    def feature_scaling(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        return X_train

    #Perform the feature scaling on test data
    def feature_scaling_test(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.transform(X_test)
        return X_test

    #Perform the feature scaling on train data using MinMaxScaler
    def feature_scaling_minmax(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        return X_train

    #Perform the feature scaling on test data using MinMaxScaler
    def feature_scaling_minmax_test(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        X_test = sc.transform(X_test)
        return X_test

    #Perform the feature scaling on train data using RobustScaler
    def feature_scaling_robust(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        X_train = sc.fit_transform(X_train)
        return X_train

    #Perform the feature scaling on test data using RobustScaler
    def feature_scaling_robust_test(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import RobustScaler
        sc = RobustScaler()
        X_test = sc.transform(X_test)
        return X_test

    #Perform the feature scaling on train data using Normalizer
    def feature_scaling_normalizer(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import Normalizer
        sc = Normalizer()
        X_train = sc.fit_transform(X_train)
        return X_train

    #Perform the feature scaling on test data using Normalizer
    def feature_scaling_normalizer_test(self,df):
        x= self.df[self.x]
        y= self.df[self.y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        from sklearn.preprocessing import Normalizer
        sc = Normalizer()
        X_test = sc.transform(X_test)
        return X_test

    #Perform all the machine learning algorithm based on regression and classification print the all the metrics and plot the graph  and return the best three model
    def model_train(self,df,x, y):
        """ Perform all the machine learning algorithm based on regression and classification print the all the metrics and plot the graph  and return the best three model """
        
        ##create a list of all the model with regreesion and classification separately
        model_list = [LinearRegression(),LogisticRegression(),DecisionTreeClassifier(),DecisionTreeRegressor(),RandomForestClassifier(),RandomForestRegressor(),KNeighborsClassifier(),KNeighborsRegressor(),SVC(),SVR(),GaussianNB()]
        model_name = ['LinearRegression','LogisticRegression','DecisionTreeClassifier','DecisionTreeRegressor','RandomForestClassifier','RandomForestRegressor','KNeighborsClassifier','KNeighborsRegressor','SVC','SVR','GaussianNB']
        model_type = ['Regression','Classification','Classification','Regression','Classification','Regression','Classification','Regression','Classification','Regression','Classification']
        model_score = []
        model_accuracy = []
        model_precision = []
        model_recall = []
        model_f1 = []
        model_auc = []
        model_brier = []
        model_mae = []
        model_mse = []
        model_rmse = []

        ##create a loop to iterate all the model fit and print the metrics and plot the graph
        x= df[x]
        y= df[y]
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = self.test_size, random_state = 30)
        for i in range(len(model_list)):
            model = model_list[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            if model_type[i] == 'Regression':
                model_score.append(model.score(X_test,y_test))
                model_accuracy.append(accuracy_score(y_test,y_pred.round()))
                model_precision.append(precision_score(y_test,y_pred.round()))
                model_recall.append(recall_score(y_test,y_pred.round()))
                model_f1.append(f1_score(y_test,y_pred.round()))
                model_auc.append(roc_auc_score(y_test,y_pred.round()))
                model_brier.append(brier_score_loss(y_test,y_pred.round()))
                model_mae.append(mean_absolute_error(y_test,y_pred.round()))
                model_mse.append(mean_squared_error(y_test,y_pred.round()))
                model_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred.round())))
                print('Model Name: ',model_name[i])
                print('Model Type: ',model_type[i])
                print('Model Score: ',model_score[i])
                print('Model Accuracy: ',model_accuracy[i])
                print('Model Precision: ',model_precision[i])
                print('Model Recall: ',model_recall[i])
                print('Model F1: ',model_f1[i])
                print('Model AUC: ',model_auc[i])
                print('Model Brier: ',model_brier[i])
                print('Model MAE: ',model_mae[i])
                print('Model MSE: ',model_mse[i])
                print('Model RMSE: ',model_rmse[i])
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')

            else:
                model_score.append(model.score(X_test,y_test))
                model_accuracy.append(accuracy_score(y_test,y_pred.round()))
                model_precision.append(precision_score(y_test,y_pred.round()))
                model_recall.append(recall_score(y_test,y_pred.round()))
                model_f1.append(f1_score(y_test,y_pred.round()))
                model_auc.append(roc_auc_score(y_test,y_pred.round()))
                model_brier.append(brier_score_loss(y_test,y_pred.round()))
                model_mae.append(mean_absolute_error(y_test,y_pred.round()))
                model_mse.append(mean_squared_error(y_test,y_pred.round()))
                model_rmse.append(np.sqrt(mean_squared_error(y_test,y_pred.round())))
                print('Model Name: ',model_name[i])
                print('Model Type: ',model_type[i])
                print('Model Score: ',model_score[i])
                print('Model Accuracy: ',model_accuracy[i])
                print('Model Precision: ',model_precision[i])
                print('Model Recall: ',model_recall[i])
                print('Model F1: ',model_f1[i])
                print('Model AUC: ',model_auc[i])
                print('Model Brier: ',model_brier[i])
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')

        ##create a dataframe of all the metrics
        model_metrics = pd.DataFrame({'Model Name':model_name,'Model Type':model_type,'Model Score':model_score,'Model Accuracy':model_accuracy,'Model Precision':model_precision,'Model Recall':model_recall,'Model F1':model_f1,'Model AUC':model_auc,'Model Brier':model_brier,'Model MAE':model_mae,'Model MSE':model_mse,'Model RMSE':model_rmse})
        model_metrics = model_metrics.sort_values(by='Model Score',ascending=False)
        print(model_metrics)

        #Create a list of top three model based on the score
        top_three_model = model_metrics['Model Name'].head(3).tolist()
        print(top_three_model)

        #roc-auc curve for all the model in one graph
        for i in range(len(model_list)):
            fpr, tpr, _ = roc_curve(y_test,y_pred)  # Replace y_true and y_pred with your actual data
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{models[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.show()

        #plot precision-recall curve for all the model
        for i in range(len(model_list)):
            model = model_list[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            if model_type[i] == 'Regression':
                pass
            else:
                plot_precision_recall_curve(model, X_test, y_test)
                plt.title(model_name[i])
                plt.show()


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
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df = imputer.fit_transform(df)


            
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


class ModelReport :



    def mlReport(self,df, x, y, model_type):
        """ fully automated report """
        #Check missing values
        check_missing = (df.isnull().sum() * 100 / df.shape[0]).sort_values(ascending=False) 
        check_missing[check_missing > 0]
        print("Missing values with percentage")
        print(check_missing)
        print()
        print()

        #Check duplicate values
        check_duplicate = df.duplicated().sum()
        
        print("Duplicate value: ",check_duplicate)
        print()
        print()

        #remove duplicate values
        df.drop_duplicates(inplace=True)
        print("Duplicate value removed")
        print()
        print()



        #Label Encoding for Object datatype all columns automatically
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        print("Handled categorical data with LabelEncoder")
        print()
        print()

        

        #handling missing value with KNNImputer
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df1 = imputer.fit_transform(df)
        df = pd.DataFrame(df1, columns = df.columns)
        print("Handled missing value with KNNImputer")
        print()
        print()

        print(df)

        #Correlation Heatmap for all columns    
        corr = df.corr()
        plt.figure(figsize=(18, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
        plt.show()

        #Remove column with correlation value more than 0.9
        corr = df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        df.drop(to_drop, axis=1, inplace=True)
        if len(to_drop) > 0:
            print("Columns dropped: ", to_drop)
        else:
            print("No columns dropped")
        print()
        print()

        #Splitting the dataset into the Training set and Test set
        x= df[x]
        y= df[y]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

       


        #regression models list
        regression_model_list = [LinearRegression(),DecisionTreeRegressor(),RandomForestRegressor(),KNeighborsRegressor(),SVR()]
        regression_model_name = ['LinearRegression','DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','SVR']

        #classification models list
        classification_model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),KNeighborsClassifier(),SVC(),GaussianNB()]
        classification_model_name = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier','SVC','GaussianNB']

        
        #if the model type is regression then perform regression model else perform classification model
        if model_type == 'Regression':
            model_list = regression_model_list
            model_name = regression_model_name
        else:
            model_list = classification_model_list
            model_name = classification_model_name

        #create a list of all the metrics
        model_score = []
        model_accuracy = []
        model_precision = []
        model_recall = []
        model_f1 = []
        model_auc = []
        model_brier = []
        model_mae = []
        model_mse = []

        #create a loop to iterate all the model fit and print the metrics and plot the graph
        for i in range(len(model_list)):
            model = model_list[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            if model_type == 'Regression':
                model_score.append(model.score(X_test,y_test))
                model_accuracy.append(accuracy_score(y_test,y_pred.round()))
                model_precision.append(precision_score(y_test,y_pred.round()))
                model_recall.append(recall_score(y_test,y_pred.round()))
                model_f1.append(f1_score(y_test,y_pred.round()))
                model_auc.append(roc_auc_score(y_test,y_pred.round()))
                model_brier.append(brier_score_loss(y_test,y_pred.round()))
                model_mae.append(mean_absolute_error(y_test,y_pred.round()))
                model_mse.append(mean_squared_error(y_test,y_pred.round()))
                print('Model Name: ',model_name[i])
                print('Model Type: ',model_type)
                print('Model Score: ',model_score[i])
                print('Model Accuracy: ',model_accuracy[i])
                print('Model Precision: ',model_precision[i])
                print('Model Recall: ',model_recall[i])
                print('Model F1: ',model_f1[i])
                print('Model AUC: ',model_auc[i])
                print('Model Brier: ',model_brier[i])
                print('Model MAE: ',model_mae[i])
                print('Model MSE: ',model_mse[i])
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')

            else:
                model_score.append(model.score(X_test,y_test))
                model_accuracy.append(accuracy_score(y_test,y_pred.round()))
                model_precision.append(precision_score(y_test,y_pred.round()))
                model_recall.append(recall_score(y_test,y_pred.round()))
                model_f1.append(f1_score(y_test,y_pred.round()))
                model_auc.append(roc_auc_score(y_test,y_pred.round()))
                model_brier.append(brier_score_loss(y_test,y_pred.round()))
                model_mae.append(mean_absolute_error(y_test,y_pred.round()))
                model_mse.append(mean_squared_error(y_test,y_pred.round()))
                print('Model Name: ',model_name[i])
                print('Model Type: ',model_type)
                print('Model Score: ',model_score[i])
                print('Model Accuracy: ',model_accuracy[i])
                print('Model Precision: ',model_precision[i])
                print('Model Recall: ',model_recall[i])
                print('Model F1: ',model_f1[i])
                print('Model AUC: ',model_auc[i])
                print('Model Brier: ',model_brier[i])
                print('Model MAE: ',model_mae[i])
                print('Model MSE: ',model_mse[i])
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')

        #create a dataframe of all the metrics
        model_metrics = pd.DataFrame({'Model Name':model_name,'Model Type':model_type,'Model Score':model_score,'Model Accuracy':model_accuracy,'Model Precision':model_precision,'Model Recall':model_recall,'Model F1':model_f1,'Model AUC':model_auc,'Model Brier':model_brier,'Model MAE':model_mae,'Model MSE':model_mse})
        model_metrics = model_metrics.sort_values(by='Model Score',ascending=False)
        print(model_metrics)

        #Create a list of top three model based on the score
        top_three_model = model_metrics['Model Name'].head(3).tolist()

        print()
        print()
        print("Top best 3 models : ",top_three_model)
        

        
        

