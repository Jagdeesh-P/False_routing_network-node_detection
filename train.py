import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('network_nodes_cleaned_with_categoricals.csv')

# Separate the target and features
X = data.drop(['Node_ID', 'Malicious_Node'], axis=1)
y = data['Malicious_Node']

# Convert categorical variables to numerical values using OneHotEncoder
categorical_columns = ['Routing_Update_Frequency', 'Route_Preference', 'Neighbor_Trustworthiness',
                       'Packet_Inspection', 'Historical_Behavior']
numerical_columns = X.columns.difference(categorical_columns)

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded_categorical = encoder.fit_transform(X[categorical_columns])
X_encoded_categorical_df = pd.DataFrame(X_encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical columns with numerical columns
X_encoded = pd.concat([X[numerical_columns].reset_index(drop=True), X_encoded_categorical_df], axis=1)

# Scale numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Feature selection using GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
selector = SelectFromModel(gbc, threshold='mean').fit(X_res, y_res)
X_selected = selector.transform(X_res)

# Save the fitted encoder
joblib.dump(encoder, 'onehot_encoder.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_res, test_size=0.2, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 5, 7, 9]
}

# Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Fit the best estimator on the training data
best_estimator.fit(X_train, y_train)

# Predict on the test data
y_pred = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the model and selector to disk
joblib.dump(best_estimator, 'gradient_boosting_model.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model, encoder, and feature selector saved to disk.")
