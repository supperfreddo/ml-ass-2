# 1. Import libraries
import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2. Set start time
start_time = time.time()

# 3. Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

# 4. Convert the power and toughness columns to numeric
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')

#### DO WE WANNA USE THIS?
# Set column power and toughness null values to 0
# df['power'] = df['power'].fillna(0)
# df['toughness'] = df['toughness'].fillna(0)

# Drop rows with missing values in power and toughness columns
df = df.dropna(subset=['power', 'toughness'])

# 5. Split data into features and target
X_numeric = df[['cmc', 'power', 'toughness']]
X_categorical = df[['layout', 'mana_cost', 'color_identity', 'type', 'supertypes', 'subtypes', 'set']]
y = df['rarity']

# 6. One-hot encode the categorical features
encoder_X = OneHotEncoder()
X_encoded = encoder_X.fit_transform(X_categorical).toarray()

# 7. Encode the categorical target
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# 8. Combine the numerical and encoded categorical features
X_combined = np.hstack((X_numeric, X_encoded))

# 9. Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_encoded, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

#### 10. Naive Bayes
print("NAIVE BAYAS:")
# Hyperparameters
alpha_multinomial = 1.0  # Smoothing parameter for Multinomial Naive Bayes
priors_gaussian = None   # Priors for Gaussian Naive Bayes (None means they will be calculated from the data)

# Train Naive Bayes classifiers
string_classifier = MultinomialNB(alpha=alpha_multinomial)
numeric_classifier = GaussianNB(priors=priors_gaussian)

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
accuracy_test = accuracy_score(y_test, combined_pred_test)
accuracy_val = accuracy_score(y_val, combined_pred_val)

print("Combined Naive Bayes Testing Accuracy:       {}%". format(accuracy_test*100))
print("Combined Naive Bayes Validation Accuracy:    {}%". format(accuracy_val*100))

#### 11. Random Forest
print("\nRANDOM FOREST:")
# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [1, 25, 50], #### Seems to always prefer the highest value
    'max_depth': [None, 10, 20, 30], #### Seems to always prefer none
    'min_samples_split': [2, 3, 4], #### Seems to always prefer 2
    'min_samples_leaf': [1, 2, 4] #### Seems to always prefer 1
}

# Print used hyperparamets for grid search
print("Used Hyperparameters for Random Forest:")
print(param_grid)

# Create random forest model
rf_model = RandomForestClassifier(random_state=0)

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by the grid search
print("Best Hyperparameters for Random Forest:")
print(grid_search.best_params_)

# Predict using the best model
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Decode the predicted labels back to original categorical form
y_val_pred_decoded = encoder_y.inverse_transform(y_val_pred)
y_val_decoded = encoder_y.inverse_transform(y_val)

# Predict using the best model on the test set
y_test_pred = best_model.predict(X_test)

# Decode the predicted labels back to original categorical form
y_test_pred_decoded = encoder_y.inverse_transform(y_test_pred)
y_test_decoded = encoder_y.inverse_transform(y_test)

# Calculate accuracy on the validation and test set
accuracy_test = accuracy_score(y_test_decoded, y_test_pred_decoded)
accuracy_val = accuracy_score(y_val_decoded, y_val_pred_decoded)

print("Random Forest Test Accuracy with Best Hyperparameters:       {}%". format(accuracy_test*100))
print("Random Forest Validation Accuracy with Best Hyperparameters: {}%". format(accuracy_val*100))

# 12. Print exucution time
print("\nExecution Time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
