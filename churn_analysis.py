import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for the beautiful heatmap style!
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# [Step 1] Load Data
df = pd.read_csv('dataset.csv') 
df = df.sort_values(by='signup_date')

# [Step 2] Split Data (Anti-Leakage)
X = df[['days_since_last_action', 'total_actions_6_months']]
y = df['churn_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# [Step 3] Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# [Step 4] Generate Chart 1: The Engagement Bar Chart
plt.figure(figsize=(6, 4))
df.groupby('churn_label')['total_actions_6_months'].mean().plot(kind='bar', color=['#4CAF50', '#FF5722'])
plt.title('Average User Engagement')
plt.xlabel('Status (0=Active, 1=Churned)')
plt.ylabel('Average Actions')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_engagement_chart.png', dpi=300)
plt.close()

# [Step 5] NEW! Generate Chart 2: Confusion Matrix Heatmap Picture
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions)
# Building a clean blue heatmap grid
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Active', 'Predicted Churn'],
            yticklabels=['Actual Active', 'Actual Churn'])
plt.title('Model Prediction Accuracy Matrix')
plt.tight_layout()

# Save the second picture asset!
plt.savefig('model_accuracy_heatmap.png', dpi=300)
print(" Successfully saved BOTH 'churn_engagement_chart.png' and 'model_accuracy_heatmap.png'!")
