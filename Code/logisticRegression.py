
from data import df

df.head()

import pandas as pd
import numpy as np
import seaborn as sns


import pandas as pd
import numpy as np
import random as rand

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from scipy.stats import chi2_contingency, boxcox

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

#we have skewness of blood sugar ~2.26 and of body temperature around 1.75, which is considered highly skewed data
#we can transform our Blood Sugar and Age columns, so they look more normally distributed
#log transformation
df_transformed = df.copy()

df_transformed["Age"] = df["Age"].apply(np.log)
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

# N * D matrix
# convert it to one hot encoded data
one_hot_encoded_data = enc.transform(df_transformed).toarray()
# randomize the data - right now is random
rand.shuffle(one_hot_encoded_data)  # since data taken in is sorted by feature

X_train, X_test, Y_train, Y_test = train_test_split(X_bias, Y_encoded, test_size=0.2, random_state=42)

#softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  #subtract constant vector to avoid overflow errors
    return exp_z/np.sum(exp_z, axis=1, keepdims=True) #calculate sofmax for all values in z matrix 

#each row represents a datapoint, each column represnts a class; a point represents the probability (0-1) that a datapoint is in that class; for row i: [e^z(i0)/e^z(i0)+e^z(i1)+e^z(i2), e^z(i1)/e^z(i0)+e^z(i1)+e^z(i2), e^z(i2)/e^z(i0)+e^z(i1)+e^z(i2)]


class logisticRegression:

    def __init__(self, learning_rate=0.01, max_iters=500, epsilon=0.0001):
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # represents termination condition (smallest objective change)
        self.max_iters = max_iters  # maximum number of iterations of gradient descent

    def one_hot_encoded (self, y, num_classes):
        #converts labels to one-hot encoding form (N x K matrix)
        Y_one_hot = np.zeros((y.shape[0], num_classes))
        Y_one_hot[np.arange(y.shape[0]), y] = 1 
        return Y_one_hot

    #train the weights
    def fit(self, x, y): 
        x = np.c_[np.ones(x.shape[0]), x] #add a column of 1's at the start of X for biases
        K = 3    #number of classes (high, low, or mid risk)
        Y = self.one_hot_encoded(y,K)
        N, D = x.shape #x is datapoints by features matrix; N is number of women tested; D is number of features
       
        self.W = np.zeros((D,K))
        gradient = np.zeros_like(self.W) #intialize matrix of 0 for gradient CE loss 
        gradient_norm = np.inf 
        iteration_num = 0
        
        #gradient descent
        while gradient_norm>self.epsilon and iteration_num<self.max_iters: 
            Z  = np.dot(x, self.W)  #x⋅W  --> (N x K matrix)
            P = softmax(Z)  #computes softmax probabilities (N x K)
            
            #gradient of the CE loss function
            gradient = np.dot(x.T, (P-Y)) / N  #1/N * X(transpose)⋅(P-Y); where (P-Y) is ŷ-y   (D x k matrix)
            
            self.W = self.W - self.learning_rate*gradient
            iteration_num+=1
            gradient_norm=np.linalg.norm(gradient) #update gradient norm (scalar value representing gradient size)
        
        return self

    #predict output for given input
    def predict (self, x):
        x = np.c_[np.ones(x.shape[0]), x] #add a column of 1's at the start of x for biases
        Z = np.dot(x, self.W)
        P = softmax(Z)
        return np.argmax(P, axis=1) #returns an array of predicted class labels of each point (by highest probability of each row/datapoint)
    
#train and test the model
model = logisticRegression(learning_rate=0.01, max_iters=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#evaluate accuracy
accuracy = np.mean(Y_pred == Y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")