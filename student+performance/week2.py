import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

# Load your Kaggle dataset
df = pd.read_csv("stf.csv")  # rename this to your actual filename

print(df.columns)

# Target: Pass/Fail based on GPA
df['Pass'] = np.where(df['GPA'] >= 2.0, 1, 0)  # adjust threshold if needed
print(df[['GPA', 'Pass']].head())

# Features (numeric)
features = ['Age', 'StudyTimeWeekly', 'Absences']
X = df[features]
y = df['Pass']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Evaluation
print("\n Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))

print("\n Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))

# Confusion Matrices
fig, ax = plt.subplots(1,2, figsize=(12,5))
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax[0], cmap='Blues')
ax[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_estimator(tree, X_test, y_test, ax=ax[1], cmap='Greens')
ax[1].set_title("Decision Tree")
plt.show()

# Feature importance
coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr.coef_[0]
})
sns.barplot(x='Coefficient', y='Feature', data=coeff_df)
plt.title("Feature Impact (Logistic Regression)")
plt.show()

plt.figure(figsize=(15,8))
plot_tree(tree, feature_names=features, class_names=["Fail", "Pass"], filled=True)
plt.show()
