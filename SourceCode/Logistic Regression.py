import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Read data
data = pd.read_csv('selected_columns.csv')

# Calculate BMI
data['BMI'] = data['Weight'] / (data['Height'] ** 2)

# Labeling obesity
def classify_obesity(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif 18.5 <= bmi < 24.9:
        return 1  # Normal
    elif 25 <= bmi < 29.9:
        return 2  # Overweight
    else:
        return 3  # Obesity

data['Obesity'] = data['BMI'].apply(classify_obesity)

# Select features and labels
features = ['Age', 'Height', 'Weight']
X = data[features]
y = data['Obesity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model Logistic Regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print result
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Draw ROC
y_prob = model.predict_proba(X_test)
plt.figure(figsize=(10, 8))

for i in range(len(model.classes_)):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Decision Boundary 
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Reduce the number of features to 2 to draw the decision boundary
if X_train.shape[1] > 2:
    X_train_2d = X_train[:, :2]
    model_2d = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model_2d.fit(X_train_2d, y_train)
    plot_decision_boundary(X_train_2d, y_train, model_2d)

