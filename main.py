import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from quad_tree_grid_search import QuadTreeGridSearch

# Prepare data
x, y = load_breast_cancer(return_X_y=True)
x = StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

# Initialize and run quad tree grid search
qtgs = QuadTreeGridSearch(
  C_range=(-8, 8),
  gamma_range=(-8, 8),
  cv=5,
  min_cv_score=0.7,
  max_sv_ratio=0.5,
  min_grid_size=0.5,
)

# Fit and get best parameters
qtgs.fit(x, y)
best_params = qtgs.get_best_params()
print('Best parameters:', best_params)

svc = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', random_state=42)

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

# Print classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Number of support vectors
print('\nNumber of support vectors:', svc.n_support_)
print('Support vector ratio:', sum(svc.n_support_) / len(x_train))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Malignant', 'Benign']
sns.heatmap(
  cm,
  annot=True,
  fmt='d',
  cmap='Blues',
  xticklabels=labels if labels else 'auto',
  yticklabels=labels if labels else 'auto',
)

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Additional model insights
print('\nModel Insights:')
print(f'Training set score: {svc.score(x_train, y_train):.3f}')
print(f'Test set score: {svc.score(x_test, y_test):.3f}')
