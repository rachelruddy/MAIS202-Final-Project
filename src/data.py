import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

#from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
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
from imblearn.over_sampling import SMOTE  # Import SMOTE

data = pd.read_csv("../data/Maternal_Health_Risk_Data_Set.csv")
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
pd.set_option('future.no_silent_downcasting', True)
df.replace({"high risk":2, "mid risk":1, "low risk":0}, inplace=True)
df['RiskLevel'] = df['RiskLevel'].astype(int)  # Explicitly cast to int

print(df.head())

#SKEWNESS
skew_limit = 0.75 # define a limit above which we will log transform

'''# Create a list of numerical colums to check for skewing
mask = data.dtypes != object
num_cols = data.columns[mask]

skew_vals = df[num_cols].skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
print(skew_cols) #we have skewness of blood sugar ~2.26 and of body temperature around 1.75, which is considered highly skewed data'''

#we can transform our Blood Sugar and Age columns, so they look more normally distributed
#log transformation
df_transformed = df.copy()

df_transformed["Age"] = df["Age"].apply(np.log)
print(df_transformed["Age"].skew())

#box cox
bc_result = boxcox(df.BS)
df_transformed['BS'] = bc_result[0]
'''boxcox_bs = pd.DataFrame(bc_result[0], columns=['BS'])
lambd = bc_result[1]
df_transformed['BS'] = boxcox_bs['BS']
df_transformed[['BS', 'Age']].skew().to_frame().rename(columns={0:'Skew'}).sort_values('Skew', ascending=False)'''
#After performing the log and boxcox transformations, the skewness of:
#It made the distributions fairly normal.

#DATA PREPROCESSING
#variable X equal to the numerical features and a variable y equal to the "RiskLevel" column.
X = df_transformed.drop('RiskLevel', axis=1).values
y = df_transformed['RiskLevel'].values

y = y.astype(int) # Ensure y is integer type BEFORE split

#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

###
###
# Check the size of train and test sets
print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Check class distribution in training and test sets
import collections
print("Class distribution in full dataset:", collections.Counter(y))
print("Class distribution in training set:", collections.Counter(y_train))
print("Class distribution in test set:", collections.Counter(y_test))
###
###


#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#apply smote 
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# X_train = X_train_resampled
# y_train = y_train_resampled
