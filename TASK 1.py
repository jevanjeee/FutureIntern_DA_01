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

#  Dropping columns
TITANIC.drop(columns=["Cabin", "Name", "Ticket"], axis=1, inplace=True) 

# Filling null values in the 'Embarked' column with the mode (most frequent value)
TITANIC["Embarked"] = TITANIC["Embarked"].fillna(TITANIC["Embarked"].mode()[0])  

# Filling missing age values with the median
TITANIC['Age'].fillna(TITANIC["Age"].median(), inplace=True)

# Confirming that there are no null values
print(TITANIC.isnull().sum())

# Outlier Removal using IQR (Interquartile Range)
# For Age
Q1_age = TITANIC['Age'].quantile(0.25)
Q3_age = TITANIC['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

# Remove outliers in Age
TITANIC = TITANIC[(TITANIC['Age'] >= lower_bound_age) & (TITANIC['Age'] <= upper_bound_age)]

# For Fare
Q1_fare = TITANIC['Fare'].quantile(0.25)
Q3_fare = TITANIC['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

# Remove outliers in Fare
TITANIC = TITANIC[(TITANIC['Fare'] >= lower_bound_fare) & (TITANIC['Fare'] <= upper_bound_fare)]

print(TITANIC)
