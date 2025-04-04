import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, boxcox
from imblearn.over_sampling import SMOTE  # Import SMOTE

# Load the dataset
data = pd.read_csv("../../data/Heart_Prediction_Quantum_Dataset.csv")

# Copy dataset
df = data.copy()
# Get rid of duplicated values
df = df.drop_duplicates().reset_index(drop=True)

# Display heatmap of correlations between features
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Feature Correlations")
# plt.show()

# Prepare data. Separate into features and targets.
X = df.drop(columns=["HeartDisease"])  
y = df["HeartDisease"]  

# Split data into train and test data using SKLearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


