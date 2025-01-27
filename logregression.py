import pandas as pd #logregression.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('game.csv')

# Drop rows with missing critical columns
df.dropna(subset=['Name', 'Developer', 'Genre'], inplace=True)

# Check if 'Release Year' exists and create it from 'Date Released'
if 'Release Year' not in df.columns:
    df['Date Released'] = pd.to_datetime(df['Date Released'], errors='coerce')
    df['Release Year'] = df['Date Released'].dt.year

# Handle missing values in numeric columns (e.g., Release Year)
numeric_cols = ['Release Year']  # Specify numeric columns that might have missing values
imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent value to impute missing values
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Encode categorical columns (e.g., Developer, Genre) using LabelEncoder
label_encoder = LabelEncoder()
df['Developer'] = label_encoder.fit_transform(df['Developer'])
df['Genre'] = label_encoder.fit_transform(df['Genre'])

# Define the target variable (classification target)
df['classification'] = df['Genre'].apply(lambda x: 'High Value' if x == 0 else ('Potential' if x == 1 else 'Low Value'))  # Example target column

# Convert classification labels to numbers
df['classification'] = df['classification'].map({'High Value': 0, 'Potential': 1, 'Low Value': 2})

# Separate features and target
X = df[['Developer', 'Genre', 'Release Year']]  # Features (adjust based on available columns)
y = df['classification']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  