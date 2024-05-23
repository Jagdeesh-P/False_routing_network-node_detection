import csv
import numpy as np
import pandas as pd

np.random.seed(42)

# Define categorical values for some columns
routing_update_frequency_categories = ['Low', 'Medium', 'High']
route_preference_categories = ['Preferred', 'Neutral', 'Not Preferred']
neighbor_trustworthiness_categories = ['Low', 'Medium', 'High']
packet_inspection_categories = ['None', 'Partial', 'Full']
historical_behavior_categories = ['Good', 'Average', 'Bad']

data = [['Node_ID', 'Routing_Update_Frequency', 'Routing_Table_Consistency', 'Route_Stability',
         'Route_Convergence_Time', 'Route_Error_Rate', 'Route_Preference', 'Neighbor_Trustworthiness',
         'Packet_Inspection', 'Historical_Behavior', 'Malicious_Node']]

for i in range(1, 4501):
    row = [i]

    # Assign categorical values
    row.append(np.random.choice(routing_update_frequency_categories))

    # Assign numerical values
    row += list(np.random.rand(4))

    # Assign categorical values
    row.append(np.random.choice(route_preference_categories))
    row.append(np.random.choice(neighbor_trustworthiness_categories))
    row.append(np.random.choice(packet_inspection_categories))
    row.append(np.random.choice(historical_behavior_categories))

    # Assign binary value for malicious node
    row.append(int(np.random.rand() > 0.5))

    # Introduce NaN values randomly in numerical columns
    if np.random.rand() < 0.1:
        nan_index = np.random.randint(2, 6)
        row[nan_index] = np.nan

    data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])

# Convert numerical columns to float and fill NaNs with column mean
numerical_columns = ['Routing_Table_Consistency', 'Route_Stability', 'Route_Convergence_Time', 'Route_Error_Rate']
for column in numerical_columns:
    df[column] = df[column].astype(float)
    df[column].fillna(df[column].mean(), inplace=True)

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Save to CSV
df.to_csv('network_nodes_cleaned_with_categoricals.csv', index=False)

print("Synthetic dataset generated, cleaned, and saved to 'network_nodes_cleaned_with_categoricals.csv'.")
