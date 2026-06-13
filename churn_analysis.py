import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Added for generating pictures!
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("==================================================")
print("  VIRTUALWORKS LABS - CUSTOMER CHURN PIPELINE     ")
print("==================================================\n")

# [Step 1] Load Data
df = pd.read_csv('dataset.csv')
df = df.sort_values(by='signup_date')

# [Step 2] Isolate Features and Split (Anti-Leakage)
X = df[['days_since_last_action', 'total_actions_6_months']]
y = df['churn_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# [Step 3] Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# [Step 4] Generate Visual Picture Chart
print("[Visual Engine] Generating churn insight chart...")
plt.figure(figsize=(8, 5))
# Create a bar chart comparing active vs churned users
df.groupby('churn_label')['total_actions_6_months'].mean().plot(kind='bar', color=['#4CAF50', '#FF5722'])
plt.title('Average User Engagement: Active vs Churned')
plt.xlabel('Customer Status (0 = Active, 1 = Churned)')
plt.ylabel('Average Actions (Past 6 Months)')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the picture to the folder so we can upload it to GitHub!
plt.savefig('churn_engagement_chart.png', dpi=300)
print("Saved chart image as 'churn_engagement_chart.png'\n")

# [Step 5] Evaluation Metrics
predictions = model.predict(X_test)
print("============ CUSTOMER CHURN MATRIX ============")
print(classification_report(y_test, predictions))
print("===============================================")
