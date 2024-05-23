from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model, encoder, scaler, and feature selector
model = joblib.load('gradient_boosting_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
selector = joblib.load('feature_selector.pkl')
scaler = joblib.load('scaler.pkl')

# Define categories as used in the training data
routing_update_categories = ['Low', 'Medium', 'High']
route_preference_categories = ['Preferred', 'Neutral', 'Not Preferred']
trustworthiness_categories = ['Low', 'Medium', 'High']
packet_inspection_categories = ['Partial', 'Full']
historical_behavior_categories = ['Good', 'Average', 'Bad']


@app.route('/')
def home():
    return render_template('index.html',
                           routing_update_categories=routing_update_categories,
                           route_preference_categories=route_preference_categories,
                           trustworthiness_categories=trustworthiness_categories,
                           packet_inspection_categories=packet_inspection_categories,
                           historical_behavior_categories=historical_behavior_categories)


@app.route('/', methods=['POST'])
def predict():
    # Get form data
    routing_update = request.form['routingUpdate']
    routing_table = float(request.form['routingTable'])
    route_stability = float(request.form['routeStability'])
    convergence_time = float(request.form['convergenceTime'])
    error_rate = float(request.form['errorRate'])
    preference = request.form['preference']
    trustworthiness = request.form['trustworthiness']
    inspection = request.form['inspection']
    behavior = request.form['behavior']

    # Create DataFrame from form data
    input_data = pd.DataFrame([[routing_update, routing_table, route_stability, convergence_time, error_rate,
                                preference, trustworthiness, inspection, behavior]],
                              columns=['Routing_Update_Frequency', 'Routing_Table_Consistency', 'Route_Stability',
                                       'Route_Convergence_Time', 'Route_Error_Rate', 'Route_Preference',
                                       'Neighbor_Trustworthiness', 'Packet_Inspection', 'Historical_Behavior'])

    # Encode categorical features
    categorical_columns = ['Routing_Update_Frequency', 'Route_Preference', 'Neighbor_Trustworthiness',
                           'Packet_Inspection', 'Historical_Behavior']
    X_encoded_categorical = encoder.transform(input_data[categorical_columns])
    X_encoded_categorical_df = pd.DataFrame(X_encoded_categorical,
                                            columns=encoder.get_feature_names_out(categorical_columns))

    # Combine encoded categorical columns with numerical columns
    numerical_columns = input_data.columns.difference(categorical_columns)
    X_encoded = pd.concat([input_data[numerical_columns].reset_index(drop=True), X_encoded_categorical_df], axis=1)

    # Scale numerical values
    X_scaled = scaler.transform(X_encoded)

    # Feature selection
    X_selected = selector.transform(X_scaled)

    # Predict using the loaded model
    prediction = model.predict(X_selected)
    prediction_label = 'Malicious and it is not safe' if prediction[0] == 1 else 'Non-Malicious and it is safe'

    return render_template('result.html', prediction=prediction_label)


if __name__ == '__main__':
    app.run(debug=True)
