import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from data import X_train, X_test, y_train, y_test  
from logisticRegression import logisticRegression  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#grid search
learning_rates = [0.5, 0.1, 0.05, 0.01,0.001]
max_iters_list = [200, 250, 500, 700, 1000]
epsilons = [1e-2, 1e-4, 1e-6]

best_params = None
best_score = 0

for lr, max_iter, eps in product(learning_rates, max_iters_list, epsilons):
    model = logisticRegression(learning_rate=lr, max_iters=max_iter, epsilon=eps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_score:
        best_score = acc
        best_params = (lr, max_iter, eps)

model = logisticRegression(learning_rate=best_params[0], max_iters=best_params[1], epsilon=best_params[2])

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
plt.show

#generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=['Low Risk', 'Mid Risk', 'High Risk'], yticklabels=['Low Risk', 'Mid Risk', 'High Risk'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



