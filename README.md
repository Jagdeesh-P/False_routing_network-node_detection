# Fraud_routing_network-node_detection

## DETECTING NETWORK NODES GENERATING FALSE ROUTING INFORMARION

### AIM
To develop a machine learning-based web application that detects malicious network nodes using gradient boosting, feature selection, and oversampling techniques, integrated with a user-friendly interface for real-time prediction and analysis. 

## Description
In modern computer networks, the presence of malicious nodes can pose significant security threats, leading to data breaches, unauthorized access, and other cyber attacks. Identifying these malicious nodes in real-time is crucial for maintaining network security and integrity. The objective of this project is to create a robust machine learning model that can accurately classify network nodes as either malicious or benign based on various features related to their behavior and characteristics. The model will be trained on a synthetic dataset with both categorical and numerical features, using techniques such as one-hot encoding, scaling, oversampling with SMOTE, and feature selection through gradient boosting. Additionally, a web application will be developed to enable users to input new network node data and receive real-time predictions regarding the node's status. This application will provide an intuitive interface for non-technical users, making it easier to monitor and manage network security.

### Accuracy: 0.5811529933481153

### Classification Report:
  precision recall f1-score support
0 0.59        0.54     0.56   463
1 0.57        0.63     0.60   439

accuracy               0.58 902
macro avg    0.58 0.58 0.58 902
weighted avg 0.58 0.58 0.58 902

