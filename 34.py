import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
data = pd.read_csv('D:/Datasets/datasets/Heart_disease_logregression.csv')

# Step 3: Data Preprocessing
print(data.head())
print(data.info())

# Handle Missing Values
data.fillna(data.mean(), inplace=True)

# Feature Selection
X = data.drop('HeartDisease', axis=1)  # Adjust 'HeartDisease' if needed
y = data['HeartDisease']                # Target variable

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create a ColumnTransformer with OneHotEncoder for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'  # Keep the other columns as they are
)

# Create a pipeline for preprocessing and model training
pipeline_log_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Multiple Classification Models
pipeline_log_reg.fit(X_train, y_train)
pipeline_dt.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred_log = pipeline_log_reg.predict(X_test)
y_pred_dt = pipeline_dt.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)

# Step 6: Evaluate Models
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(classification_report(y_test, y_pred_log))

print("Decision Tree Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print(classification_report(y_test, y_pred_dt))

print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test, y_pred_log, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(y_test, y_pred_dt, 'Decision Tree Confusion Matrix')
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest Confusion Matrix')