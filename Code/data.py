import pandas as pd
import numpy as np
import random as rand

#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

#from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier

#import xgboost as xgb

#import scipy.stats as stats 
from scipy.stats import chi2_contingency, boxcox
#from statsmodels.formula.api import ols
#from statsmodels.stats.anova import anova_lm

def split_data(data, split):
    # transpose x once to make it D * N, and take the first 18 rows ( all 6 features ). Transpose again to make it back into the model we want
    # x - finally is N * D
    x = np.transpose(np.transpose(data)[:18])

    # transpose y once, because they want y in as an array [1,2,3,4...]
    # we only to take one of the output features, we only a one dimensional ouptut with 2 possible qualitiative values.
    #  Col [18]: NB | Col [19]: B. Take Col 18 - one collum from a D * N matrix is a 1 * N matrix
    y = np.transpose(data)[18]

    split_index = int(x.shape[0] * split)
    training_set_x = x[:split_index]
    test_set_x = x[split_index:]
    training_set_y = y[:split_index]
    test_set_y = y[split_index:]

    return training_set_x, test_set_x, training_set_y, test_set_y, split_index


data = pd.read_csv("Maternal Health Risk Data Set.csv")
print(data.info())

#check for duplicates
print(data[data.duplicated(keep='first')])

print(data.describe())

#DATA CLEANING
# create copy of the dataset
df = data.copy()
#get rid of duplicated values
df = df.drop_duplicates().reset_index(drop=True)
#there is an outlier with heart rate 7 which doesn't make any sense s
#so we will replace it with the mode of the column
df.loc[df.HeartRate == 7, "HeartRate"] = 70

#FEATURE ENCODING
#perform a label encoding, where values are manually assigned to the corresponding keys
df.replace({"high risk":2, "mid risk":1, "low risk":0}, inplace=True)
print(df.head())

#SKEWNESS
skew_limit = 0.75 # define a limit above which we will log transform

# Create a list of numerical colums to check for skewing
mask = data.dtypes != object
num_cols = data.columns[mask]

skew_vals = df[num_cols].skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

print(skew_cols) #we have skewness of blood sugar ~2.26 and of body temperature around 1.75, which is considered highly skewed data
#we can transform our Blood Sugar and Age columns, so they look more normally distributed
#log transformation
df_transformed = df.copy()

df_transformed["Age"] = df["Age"].apply(np.log)
print(df_transformed["Age"].skew())

#box cox
bc_result = boxcox(df.BS)
boxcox_bs = pd.DataFrame(bc_result[0], columns=['BS'])
lambd = bc_result[1]
df_transformed['BS'] = boxcox_bs['BS']
df_transformed[['BS', 'Age']].skew().to_frame().rename(columns={0:'Skew'}).sort_values('Skew', ascending=False)
#After performing the log and boxcox transformations, the skewness of:
#It made the distributions fairly normal.

#DATA PREPROCESSING
#variable X equal to the numerical features and a variable y equal to the "RiskLevel" column.
X = df_transformed.drop('RiskLevel', axis=1)
y = df_transformed['RiskLevel']

#feature scaling
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# create the encoder and encode the data
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_transformed)
print("Encoding categories:")
print(enc.categories_)  # shows the one-hot encoding format for the data

# N * D matrix
# convert it to one hot encoded data
one_hot_encoded_data = enc.transform(df_transformed).toarray()
# randomize the data - right now is random
rand.shuffle(one_hot_encoded_data)  # since data taken in is sorted by feature
print("\nVisualized one hot encoded data, for first row:")
print(one_hot_encoded_data[0]) # gives a list of 20 features ( 6 * 3 for the features, 2 for the output features ( B or NB ))

X_train, X_test, Y_train, Y_test = train_test_split(X_bias, Y_encoded, test_size=0.2, random_state=42)