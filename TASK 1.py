# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Load CSV file
TITANIC = pd.read_csv('train.csv')

# Viewing the first rows of the dataset
print(TITANIC.head())

# Viewing the last rows of the dataset
print(TITANIC.tail())

# Viewing a random line
print(TITANIC.sample())

# Overview of the data
print(TITANIC.info())

# Checking columns with missing values
print(TITANIC.isnull().sum())

# Heatmap to show the missing values in each column
sns.heatmap(TITANIC.isnull(), yticklabels=False, cmap="viridis", cbar=False)

#  show the plot
plt.show()  

# Dropping columns
TITANIC.drop(columns=["Cabin", "Name", "Ticket"], axis=1, inplace=True) 

# Filling null values in the 'Embarked' column with the mode (most frequent value)
TITANIC["Embarked"] = TITANIC["Embarked"].fillna(TITANIC["Embarked"].mode()[0])  

# Filling missing age values with the median
TITANIC['Age'].fillna(TITANIC["Age"].median(), inplace=True)

# Confirming that there are no null values
print(TITANIC.isnull().sum())