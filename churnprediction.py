import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ChurnPredictor:
    def __init__(self, data):
        """
        Initialize with customer data
        """
        self.data = data
        self.model = None
    
    def preprocess_data(self):
        """
        Preprocess data by encoding categorical variables and splitting into train/test sets
        """
        # Convert 'Contract_Type' from categorical to one-hot encoded variables
        self.data = pd.get_dummies(self.data, columns=['Contract_Type'], drop_first=True)

        # Features: everything except 'Customer_ID' and 'Churn_Flag'
        X = self.data.drop(columns=['Customer_ID', 'Churn_Flag'])
        y = self.data['Churn_Flag']

        # Split the data into training and testing sets
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, model_type='LogisticRegression'):
        """
        Train the model based on user-selected type (LogisticRegression or RandomForest)
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()

        # Select model based on input
        if model_type == 'LogisticRegression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier()

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Evaluate the model performance
        self.evaluate_model(y_test, y_pred)

    def evaluate_model(self, y_test, y_pred):
        """
        Evaluate the model using accuracy, precision, and recall
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Model Performance: Accuracy={accuracy}, Precision={precision}, Recall={recall}")

    def predict_churn(self, new_customer):
        """
        Predict churn for a new customer (input as a list of features)
        """
        return self.model.predict([new_customer])[0]

    def retention_rate(self):
        """
        Calculate customer retention rate (percentage of customers who did not churn)
        """
        retention = (self.data['Churn_Flag'] == 0).mean()
        print(f"Customer Retention Rate: {retention * 100:.2f}%")
        return retention
