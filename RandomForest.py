# Import necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset (replace 'MedicalData.csv' with your dataset file path)
data = pd.read_csv('MedicalData.csv')

# Initialize label encoders and scaler
label_encoders = {
    'sex': LabelEncoder(),
    'region': LabelEncoder()
}

# Encode categorical variables
data['sex'] = label_encoders['sex'].fit_transform(data['sex'])  # Male = 0, Female = 1
data['region'] = label_encoders['region'].fit_transform(data['region'])  # Encode region

# Initialize the scaler
scaler = StandardScaler()

# Scale the numerical features
data[['age', 'bmi', 'charges']] = scaler.fit_transform(data[['age', 'bmi', 'charges']])

# Split the data into features (X) and target (y)
X = data.drop('smoker', axis=1)  # All columns except 'smoker' are features
y = data['smoker']  # 'smoker' is the target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)  # You can replace this with any other model
model.fit(X_train, y_train)

# Function to classify if a patient is a smoker
def classify_smoker(patient_data):
    """
    Predict if the patient is a smoker or not.
    Args:
        patient_data (dict): A dictionary containing patient details:
            - age
            - sex ('male' or 'female')
            - bmi
            - children (number of dependents)
            - region ('northeast', 'northwest', 'southeast', 'southwest')
            - charges
    Returns:
        str: 'Smoker' or 'Non-Smoker'
    """
    # Prepare input data
    df = pd.DataFrame([patient_data])
    df['sex'] = label_encoders['sex'].transform(df['sex'])
    df['region'] = label_encoders['region'].transform(df['region'])
    df[['age', 'bmi', 'charges']] = scaler.transform(df[['age', 'bmi', 'charges']])  # Apply scaling to numeric columns

    # Predict smoker status
    prediction = model.predict(df)
    return "Smoker" if prediction[0] == 1 else "Non-Smoker"

# Example usage:
patient_example = {
    'age': 30,
    'sex': 'female',
    'bmi': 25.5,
    'children': 1,
    'region': 'southwest',
    'charges': 3200.0
}

# Classify the patient
result = classify_smoker(patient_example)
print("\nPatient Classification:", result)

# Output the number of smokers and non-smokers in the dataset
num_smokers = y.value_counts().get(1, 0)  # 1 represents smokers
num_non_smokers = y.value_counts().get(0, 0)  # 0 represents non-smokers

print(f"\nTotal number of smokers in the dataset: {num_smokers}")
print(f"Total number of non-smokers in the dataset: {num_non_smokers}")

# Evaluate the model's performance on the test set
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting the Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Smoker", "Smoker"], yticklabels=["Non-Smoker", "Smoker"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Plotting the feature importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Plotting the distribution of smokers and non-smokers
plt.figure(figsize=(6, 4))
sns.countplot(x='smoker', data=data, palette='pastel')
plt.title('Distribution of Smokers and Non-Smokers')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Smoker', 'Smoker'])
plt.show()

