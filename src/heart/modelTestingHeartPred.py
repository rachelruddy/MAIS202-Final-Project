import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from heartData import X_train, X_test, y_train, y_test  
from logRegHeartDisease import logRegHeartDisease  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# define hyperparameter search space
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
max_iters_list = [500, 1000, 1500, 2000]
epsilons = [1e-3, 1e-5, 1e-7]

# initialize cross-validation (stratified k-fold)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

best_params = None
best_score = 0

# grid search implemented with cross-validation
for lr, max_iter, eps in product(learning_rates, max_iters_list, epsilons):
    fold_accuracies = []  
    
    # perform cross-validation
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # apply SMOTE inside the fold
        smote = SMOTE(random_state=42)
        X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

        # train and evaluate model
        model = logRegHeartDisease(learning_rate=lr, max_iters=max_iter, epsilon=eps)
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        fold_accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
    
    #compute average accuracy
    avg_acc = sum(fold_accuracies) / len(fold_accuracies)

    # update best parameters
    if avg_acc > best_score:
        best_score = avg_acc
        best_params = (lr, max_iter, eps)

# train the final model using the best hyperparameters on the full training set
print(f"Best Hyperparameters: LR={best_params[0]}, Max Iters={best_params[1]}, Epsilon={best_params[2]}")
model = logRegHeartDisease(learning_rate=best_params[0], max_iters=best_params[1], epsilon=best_params[2])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

#generate pie chart for accuracy
accuracy = accuracy_score(y_test, y_pred)
labels = ['Correct predictions', 'Incorrect predictions']
sizes = [accuracy, 1-accuracy]
colors = ['#f19ef7', '#fcffa8']
plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Model Accuracy')
plt.show()

#generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



