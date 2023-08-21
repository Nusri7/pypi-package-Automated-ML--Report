import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_cumulative_gain
from sklearn.metrics import plot_lift_curve
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
        self.df = pd.DataFrame(df)
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
    def model_train(self,df):
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
        x= self.df[self.x]
        y= self.df[self.y]
        
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



        

        
        

        








            
        

    
    


    
