import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

class EDA:
    def __init__(self, df):
        self.df = df,
        
        
    ## describe the numerical variables
    def describe(self):
        print(df.describe())

    #display categoraical variables and numerical variables
    def info(self):
        print(df.info())

    ## check missing values
    def isna(self):
        print(df.isnull().sum().sort_values(ascending=False))

    ## check duplicate values
    def duplicate_value(self):
        print(df.duplicated().sum())

    #Categoaical variables and their unique values
    def unique_categories(self):
        for col in df.columns:
            print(col, df[col].nunique())
    ## separate numerical and categorical variables
    def separate(self):
        int_vars = df.select_dtypes(include=['int',"float"]).columns.tolist() 
        cat_vars = df.select_dtypes(include='object').columns.tolist()
        return int_vars,cat_vars

    ## Figsize of the plot
    def figsize(self):
        int_vars,cat_vars = self.separate()
        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3 # To make sure there are enough rows for the s
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])
        return fig,axs

        # Adjust spacing between subplots
        fig.tight_layout()

    ## plot the histogram of the numerical variables one by one
    def hist(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            df[var].plot.hist(ax=axs[i])
            axs[i].set_title(var)

        # Show plot
        plt.show() 
            
    ## plot the scatter plot of the variables
    def scatter(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            df.plot.scatter(x=var, ax=axs[i])
            axs[i].set_title(var) 
        # Show plot
        plt.show() 

    
    ## Distribution Plot 
    def dist(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.distplot(x=df[var], ax=axs[i])
            axs[i].set_title(var) 
        # Show plot
        plt.show() 

    ## Box Plot 
    def box(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.boxplot(x=df[var], data=df, ax=axs[i])
            axs[i].set_title(var)  
        # Show plot
        plt.show() 

    ## Counter Plot for categorical variables
    def counter(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(cat_vars):
            sns.countplot(x=var, data=df, ax=axs[i]) 
            axs[i].set_title(var) 
            
        # Show plot
        plt.show()
        
    ## histogram for each integer variable with hue='Attrition’ 
    def hist_hue(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.histplot(data=df, x=var, kde=True, ax=axs[i])
            axs[i].set_title(var) 
 
        # Show plot
        plt.show()
        
    ## violin plots for numeric columns 
    def violin(self):
        int_vars,cat_vars = self.separate()
        fig, axs = self.figsize()
        # Create a histogram for each integer variable
        for i, var in enumerate(int_vars):
            sns.violinplot(x=var, data=df, ax=axs[i])
            axs[i].set_title(var) 
        # Show plot
        plt.show()