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

# Split data into features and target
types = df['type'].unique()
X_numeric = df[['cmc']]
X_categorical = df[['type']]
y = df['rarity']

# One-hot encode the categorical features
encoder_X = OneHotEncoder()
X_encoded = encoder_X.fit_transform(X_categorical).toarray()

# Encode the categorical target
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# Combine the numerical and encoded categorical features
X_combined = np.hstack((X_numeric, X_encoded))

#### Naive Bayes
print("NAIVE BAYAS:")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifiers for each feature
string_clf = MultinomialNB()
numeric_clf = GaussianNB()

string_clf.fit(X_train[:, 0].reshape(-1, 1), y_train)
numeric_clf.fit(X_train[:, 1].reshape(-1, 1), y_train)

# Predict using the trained classifiers
string_pred = string_clf.predict(X_test[:, 0].reshape(-1, 1))
numeric_pred = numeric_clf.predict(X_test[:, 1].reshape(-1, 1))

# Combine predictions from both classifiers by taking the mode
combined_pred = np.unique([string_pred, numeric_pred], axis=0)[0]

# Flatten the combined predictions
combined_pred = np.squeeze(combined_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, combined_pred)
print("Combined Naive Bayes Accuracy:", accuracy)

#### Random Forest
print("\nRANDOM FOREST:")
# Combine the numerical and encoded categorical features
X_combined = np.hstack((X_numeric, X_encoded))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=0)

# Create and fit the random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = rf_model.predict(X_test)

# Decode the predicted labels back to original categorical form
y_pred_decoded = encoder_y.inverse_transform(y_pred)
y_test_decoded = encoder_y.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print("Random Forest Accuracy:", accuracy)