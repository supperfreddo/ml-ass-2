# Import libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

# Convert the power and toughness columns to numeric
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')

# Drop rows with missing values in power and toughness columns
df = df.dropna(subset=['power', 'toughness'])

# Split data into features and target
X_numeric = df[['cmc', 'power', 'toughness']]
X_categorical = df[['type', 'set']]
y = df['rarity']

# One-hot encode the categorical features
encoder_X = OneHotEncoder()
X_encoded = encoder_X.fit_transform(X_categorical).toarray()

# Encode the categorical target
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# Combine the numerical and encoded categorical features
X_combined = np.hstack((X_numeric, X_encoded))

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_encoded, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

#### Naive Bayes
print("NAIVE BAYAS:")
# Train Naive Bayes classifiers
string_classifier = MultinomialNB()
numeric_classifier = GaussianNB()

string_classifier.fit(X_train[:, 0].reshape(-1, 1), y_train)
numeric_classifier.fit(X_train[:, 1].reshape(-1, 1), y_train)

# Predict using the trained classifiers for validation set
string_pred_val = string_classifier.predict(X_val[:, 0].reshape(-1, 1))
numeric_pred_val = numeric_classifier.predict(X_val[:, 1].reshape(-1, 1))

# Combine predictions from both classifiers for validation set by taking the mode
combined_pred_val = np.unique([string_pred_val, numeric_pred_val], axis=0)[0]
combined_pred_val = np.squeeze(combined_pred_val)

# Predict using the trained classifiers for testing set
string_pred_test = string_classifier.predict(X_test[:, 0].reshape(-1, 1))
numeric_pred_test = numeric_classifier.predict(X_test[:, 1].reshape(-1, 1))

# Combine predictions from both classifiers for testing set by taking the mode
combined_pred_test = np.unique([string_pred_test, numeric_pred_test], axis=0)[0]
combined_pred_test = np.squeeze(combined_pred_test)

# Calculate accuracy for validation and testing sets
accuracy_val = accuracy_score(y_val, combined_pred_val)
accuracy_test = accuracy_score(y_test, combined_pred_test)

print("Combined Naive Bayes Validation Accuracy:", accuracy_val)
print("Combined Naive Bayes Testing Accuracy:", accuracy_test)

#### Random Forest
print("\nRANDOM FOREST:")
# Create and fit the random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(X_train, y_train)

# Predict using the trained model on the validation set
y_val_pred = rf_model.predict(X_val)

# Decode the predicted labels back to original categorical form
y_val_pred_decoded = encoder_y.inverse_transform(y_val_pred)
y_val_decoded = encoder_y.inverse_transform(y_val)

# Predict using the trained model on the test set
y_test_pred = rf_model.predict(X_test)

# Decode the predicted labels back to original categorical form
y_test_pred_decoded = encoder_y.inverse_transform(y_test_pred)
y_test_decoded = encoder_y.inverse_transform(y_test)

# Calculate accuracy for validation and testing sets
accuracy_val = accuracy_score(y_val_decoded, y_val_pred_decoded)
accuracy_test = accuracy_score(y_test_decoded, y_test_pred_decoded)

print("Random Forest Validation Accuracy:", accuracy_val)
print("Random Forest Testing Accuracy:", accuracy_test)