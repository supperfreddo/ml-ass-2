# Import libs
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

#### Random Forest

# Generate some example data
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

# Create and fit the random forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=0) # n_estimators is the number of trees, the more the longer the model takes to run
rf_model.fit(X_combined, y_encoded)

# Create a test example
test_example_numeric = [[30]]
test_example_categorical = [['Legendary Creature — Goblin']] 

# One-hot encode the categorical features
test_example_categorical_encoded = encoder_X.transform(test_example_categorical).toarray()

# Combine the numerical and encoded categorical features
test_example_combined = np.hstack((test_example_numeric, test_example_categorical_encoded))

# Predict using the trained model
predicted_y_encoded = rf_model.predict(test_example_combined)
predicted_y = encoder_y.inverse_transform(predicted_y_encoded)
print("Predicted rarity for test_example:", predicted_y[0])



#### Naive Bayes
# # split data into features and target
# X = df[['power', 'toughness', 'cmc']]
# y = df['rarity']

# # build a Naïve Bayes model
# clf = GaussianNB()
# clf.fit(X.values, y)

# # create test example
# test_example = [[2, 1, 30]]

# # use the model to predict new example
# predicted = clf.predict(test_example)
# print(predicted)