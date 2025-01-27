import pandas as pd #forestclass.py
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('game.csv')

# Fill missing values in categorical columns with the most frequent value
categorical_columns = ['Developer', 'Producer', 'Genre', 'Operating System']
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

# Convert 'Date Released' to datetime and extract year for modeling
df['Release Year'] = pd.to_datetime(df['Date Released'], errors='coerce').dt.year

# Fill missing values in 'Release Year' with the median year
df['Release Year'].fillna(df['Release Year'].median(), inplace=True)

# Label encoding for categorical features
label_encoders = {}
for column in ['Developer', 'Producer', 'Genre', 'Operating System']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Classification function
def classify_game(row):
    if row['Genre'] in [0, 1]:  
        return 'High Value'
    elif row['Developer'] in [0, 1, 2]:  
        return 'Potential'
    elif row['Release Year'] < 2010:
        return 'At Risk'
    else:
        return 'Low Value'

# Apply classification
df['classification'] = df.apply(classify_game, axis=1)

# Map classification labels to integers
classification_map = {'High Value': 0, 'Potential': 1, 'At Risk': 2, 'Low Value': 3}
df['classification'] = df['classification'].map(classification_map)

# Features and target
X = df[['Developer', 'Producer', 'Genre', 'Operating System', 'Release Year']]
y = df['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'-'*30}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'-'*30}\n")

print("Unique classes in y_pred:", set(y_pred))
print("Unique classes in y_test:", set(y_test))

target_names = ['High Value', 'Potential', 'At Risk', 'Low Value']
if len(set(y_pred)) < 4:  
    print("Warning: Model only predicted fewer classes. Adjusting target names.")
    target_names = [name for i, name in enumerate(target_names) if i in set(y_pred)]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
print(f"{'-'*30}")

# Predicted classifications DataFrame
predicted_classifications = pd.DataFrame({'Index': X_test.index, 'Predicted': y_pred})
predicted_classifications.reset_index(drop=True, inplace=True)
print(predicted_classifications.head())  

# Prediction for new data
new_data = {'Developer': 2, 'Producer': 1, 'Genre': 0, 'Operating System': 3, 'Release Year': 2015}
new_data_df = pd.DataFrame([new_data])
prediction = model.predict(new_data_df)
predicted_class = [key for key, value in classification_map.items() if value == prediction[0]]
print(f"\nPredicted Classification for New Data: {predicted_class[0]}")