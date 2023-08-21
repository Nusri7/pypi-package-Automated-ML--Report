<h1>pypi-package-Automated-ML--Report </h1><br>

![Screenshot (92)](https://github.com/Nusri7/pypi-package-Automated-ML--Report/assets/91601996/aaf7e223-0dde-44f4-b73a-03648afe5ddf)


# Automated-ML-Report

Author : Ahamed Nusri <br>
email  : 321nusri@gmail.com <br>
GitHub : https://github.com/Nusri7 <br>

![PyPI Version](https://img.shields.io/pypi/v/eda-report.svg)
![License](https://img.shields.io/pypi/l/eda-report.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eda-report.svg)

Automated-ML-Report is a Python package designed to simplify the process of generating automated Exploratory data analysis and machine learning reports. It aims to provide a user-friendly interface for creating insightful reports based on the results of various automated machine-learning experiments.

## Installation

You can install Automated-ML-Report using `pip`:

```
pip install -i https://test.pypi.org/simple/ Automated-ML-Report==1.0.0
```

## Features
<ul>
  <li>Streamline the process of creating reports for automated machine-learning experiments.</li>
  <li>Generate detailed and visually appealing reports to communicate experiment results effectively.</li>
  <li>Customize report content, style, and format to suit your needs.</li>
</ul>


## Quick Start
Import the necessary modules:

```
from Automated-ML-Report import ModelReport
```

## Create an instance of AutomatedMLReport by providing experiment data:

```
report = ModelReport.ModelReport()
```

## Generate the automated ML report:
```
report.mlReport(df,x,y, model_type)
```
## Usage
The following example demonstrates a basic usage scenario:
```
from Automated-ML-Report import ModelReport

# Creating an object from ModelReport class
report = ModelReport.ModelReport()

# Calling the mlReport function
report.mlReport(df,["Embarked","Parch","SibSp","Age","Sex","Pclass"],"Survived", "Classification")

```

# EDA Report

EDA Report is a Python package that simplifies the process of creating exploratory data analysis (EDA) reports. It offers an intuitive way to generate comprehensive reports for understanding and visualizing dataset characteristics.

## Installation
You can install EDA Report using pip:
```
pip install -i https://test.pypi.org/simple/ Automated-ML-Report==1.0.0
```


## Features

<ul>
  <li>Streamline the process of creating reports for automated machine-learning experiments.</li>
  <li>Generate detailed and visually appealing reports to communicate experiment results effectively.</li>
  <li>Customize report content, style, and format to suit your needs.</li>
</ul>


## Quick Start
Import the necessary modules:

```
from Automated-ML-Report import EDAReport
```

## Create an instance of AutomatedMLReport by providing experiment data:

```
report = EDAReport.EDA()
```

## Generate the automated ML report:
```
report.eda_report(df,y)
```
## Usage
The following example demonstrates a basic usage scenario:
```
from Automated-ML-Report import EDAReport

# Load experiment data (replace with your data loading code)
report = EDAReport.EDA()

# Calling the mlReport function
report.edaReport(df,"Survived")

```
For more advanced usage and customization options, please refer to the documentation.

## Documentation
For detailed information about how to use Automated-ML-Report, refer to the documentation. Documentation will be uploaded soon.

## License
This project is licensed under the MIT License.

## Contact
If you have any questions, suggestions, or feedback, please contact us at your 321nusri@gmail.com.




