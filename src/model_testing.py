import seaborn as sns
import matplotlib.pyplot as plt
from data import X_train, X_test, y_train, y_test  
from logisticRegression import logisticRegression  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = logisticRegression(learning_rate=0.01, max_iters=500, epsilon=0.0001)

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


