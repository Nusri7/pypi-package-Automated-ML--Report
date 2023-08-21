import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class EDA:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
        
        
    ## describe the numerical variables
    def describe(self, df):
        print(df.describe())

    #display categoraical variables and numerical variables
    def info(self,df):
        print(df.info())

    ## check missing values
    def isna(self,df):
        print(df.isnull().sum().sort_values(ascending=False))

    ## check duplicate values
    def duplicate_value(self,df):
        print(df.duplicated().sum())

    #Categoaical variables and their unique values
    def unique_categories(self,df):
        for col in df.columns:
            print(col, df[col].nunique())
    ## separate numerical and categorical variables
    def separate(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        return int_vars,cat_vars


        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])
        return fig,axs

        # Adjust spacing between subplots
        fig.tight_layout()

    ## plot the histogram of the numerical variables one by one
    def hist(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            df[var].plot.hist(ax=axs[i],color = 'red')
            axs[i].set_title(var)

        # Show plot
        plt.show() 
            
    ## plot the scatter plot of the variables
    def scatter(self,df,y):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            df.plot.scatter(x=var, y = y, ax=axs[i])
            axs[i].set_title(var) 
        # Show plot
        plt.show() 

    
    ## Distribution Plot 
    def dist(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.displot(data=df, x=var, kde=True, ax=axs[i],color = 'yellow')
            axs[i].set_title(var) 
        # Show plot
        plt.show() 

    ## Box Plot 
    def box(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.boxplot(x=df[var], data=df, ax=axs[i],color = 'indigo')
            axs[i].set_title(var)  
        # Show plot
        plt.show() 

    ## Counter Plot for categorical variables
    def counter(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(cat_vars):
            sns.countplot(x=var, data=df, ax=axs[i]) 
            axs[i].set_title(var) 
            
        # Show plot
        plt.show()
        
    ## histogram for each integer variable with hue='Attrition’ 
    def hist_hue(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.histplot(data=df, x=var, kde=True, ax=axs[i])
            axs[i].set_title(var) 
 
        # Show plot
        plt.show()
        
    ## violin plots for numeric columns 
    def violin(self,df):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.violinplot(x=var, data=df, ax=axs[i],color = 'cyan')
            axs[i].set_title(var) 
        # Show plot
        plt.show()

    #plot the correlation matrix
    def corr(self,df):
        corr = df.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()

    #plot the distribution of the target variable
    def target(self,df):
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Attrition', data=df)
        plt.show()

    
    

    #plot the distribution of the target variable with respect to each categorical variable
    def target_cat(self,df):
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()
        fig.tight_layout()
        for i, var in enumerate(cat_vars):
            sns.countplot(x=var, data=df, ax=axs[i])
            axs[i].set_title(var)
        plt.show()


class EDAReport:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    
    #Fully automated report for EDA and save it as pdf
    def edaReport(self,df,y):
        """ fully automated report """
        #EDA
        eda = EDA()
        print("----------Info----------")
        eda.info(df)
        print()
        print()
        print("----------Unique categories----------")
        eda.unique_categories(df)
        print()
        print()
        print("----------Missing values----------")
        eda.isna(df)
        print()
        print()
        print("----------Duplicate values----------")
        eda.duplicate_value(df)
        print()
        print()
        print("----------Statical describe of numerical variables----------")
        eda.describe(df)
        print()
        print()
        print("----------Correlation----------")
        eda.corr(df)
        print()
        print()
        print("----------Histogram----------")
        eda.hist(df)
        print()
        print()
        print("----------Scatterplot----------")
        eda.scatter(df,y)
        print()
        print()
        print("----------Boxplot----------")
        eda.box(df)
        print()
        print()
        print("----------Counterplot----------")
        eda.counter(df)
        print()
        print()
        print("----------histplot----------")
        eda.hist_hue(df)
        print()
        print()
        print("----------violinplot----------")
        eda.violin(df)

    
        


