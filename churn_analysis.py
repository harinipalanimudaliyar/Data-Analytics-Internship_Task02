import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files  # Special tool to force download buttons!

# [Step 1] Load your dataset
df = pd.read_csv('dataset.csv') 
df = df.sort_values(by='signup_date')

X = df[['days_since_last_action', 'total_actions_6_months']]
y = df['churn_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# =========================================================================
# FORCE VISUALIZATION DIRECTLY ON SCREEN
# =========================================================================

# --- CHART 1: The Green & Orange Engagement Bar Chart ---
print("\n--- DRAWING CHART 1: ENGAGEMENT ---")
plt.figure(figsize=(6, 4))
df.groupby('churn_label')['total_actions_6_months'].mean().plot(kind='bar', color=['#4CAF50', '#FF5722'])
plt.title('Average User Engagement')
plt.xlabel('Status (0=Active, 1=Churned)')
plt.ylabel('Average Actions')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('churn_engagement_chart.png', dpi=300)
plt.show()  # Forces the chart to draw directly on your screen!

# --- CHART 2: The Blue Confusion Matrix Heatmap Chart ---
print("\n--- DRAWING CHART 2: ACCURACY MATRIX ---")
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Active', 'Predicted Churn'],
            yticklabels=['Actual Active', 'Actual Churn'])
plt.title('Model Prediction Accuracy Matrix')
plt.tight_layout()
plt.savefig('model_analysis_heatmap.png', dpi=300)
plt.show()  # Forces the chart to draw directly on your screen!

# =========================================================================
# AUTOMATIC LAPTOP DOWNLOAD POPUP
# =========================================================================
print("\n--- TRIGGERING DOWNLOAD LINKS TO YOUR LAPTOP ---")
files.download('churn_engagement_chart.png')
files.download('model_analysis_heatmap.png')
