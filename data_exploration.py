# data_exploration.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(url, names=columns)

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Check the DataFrame shape
print("\nShape of the DataFrame:")
print(data.shape)

# Visualize the data
try:
    sns.pairplot(data, hue="species")
    plt.show()
except ValueError as e:
    print(f"Error while plotting pairplot: {e}")
