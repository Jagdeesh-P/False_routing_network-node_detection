import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the trained model and feature selector
loaded_model = joblib.load('gradient_boosting_model.pkl')
selector = joblib.load('feature_selector.pkl')

# Define sample input data
sample_inputs = np.array([
    [0.2, 0.3, 0.4, 0.5, 'Low', 'Preferred', 'High', 'Full', 'Good'],
    [0.3, 0.4, 0.5, 0.6, 'Low', 'Neutral', 'Medium', 'Partial', 'Average'],
    [0.4, 0.5, 0.6, 0.7, 'High', 'Not Preferred', 'Low', 'None', 'Bad'],
    [0.5, 0.6, 0.7, 0.8, 'Low', 'Neutral', 'High', 'Full', 'Good'],
    [0.1, 0.2, 0.3, 0.4, 'Medium', 'Preferred', 'Low', 'Partial', 'Bad']
])

# Define column names
columns = ['Routing_Table_Consistency', 'Route_Stability', 'Route_Convergence_Time', 'Route_Error_Rate',
           'Routing_Update_Frequency', 'Route_Preference', 'Neighbor_Trustworthiness',
           'Packet_Inspection', 'Historical_Behavior']

# Convert sample input data to DataFrame
sample_df = pd.DataFrame(sample_inputs, columns=columns)

# Convert categorical variables to numerical values using OneHotEncoder
categorical_columns = ['Routing_Update_Frequency', 'Route_Preference', 'Neighbor_Trustworthiness',
                       'Packet_Inspection', 'Historical_Behavior']
numerical_columns = sample_df.columns.difference(categorical_columns)

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first')
X_encoded_categorical = encoder.fit_transform(sample_df[categorical_columns])
X_encoded_categorical_df = pd.DataFrame(X_encoded_categorical.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded categorical columns with numerical columns
sample_df_numerical = sample_df[numerical_columns].astype(float)
X_encoded = pd.concat([sample_df_numerical.reset_index(drop=True), X_encoded_categorical_df], axis=1)

# Scale numerical values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Apply the same feature selection
X_selected = selector.transform(X_scaled)

# Make predictions
predictions = loaded_model.predict(X_selected)

# Output predictions
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {'Malicious' if pred == 1 else 'Benign'}")
