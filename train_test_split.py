import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ChurnPredictor:
    def _init_(self, data):
        
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_data(self):
        
        # Convert Contract_Type into numerical values
        self.data['Contract_Type'] = self.data['Contract_Type'].map({
            'Month-to-Month': 0, 'One-Year': 1, 'Two-Year': 2
        })
        
        # Features: Tenure, Contract_Type, Monthly_Charges
        X = self.data[['Tenure', 'Contract_Type', 'Monthly_Charges']]
        # Target: Churn_Flag (0 = No Churn, 1 = Churn)
        y = self.data['Churn_Flag']
        
        return X, y

    def split_data(self, X, y):
        
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, X_train, y_train):
        
        self.model.fit(X_train, y_train)

    def predict_churn(self, X_test):
        
        return self.model.predict(X_test)

    def evaluate_performance(self, y_test, y_pred):
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        return accuracy, precision, recall

    def calculate_churn_probability(self, X_test):
        
        return self.model.predict_proba(X_test)[:, 1]

    def calculate_retention_rate(self):
        
        total_customers = len(self.data)
        churned_customers = self.data['Churn_Flag'].sum()
        retention_rate = (total_customers - churned_customers) / total_customers
        return retention_rate

# Main function to demonstrate the usage of the ChurnPredictor class
if __name__== "_main_":
    # Load the dataset (replace with the path to your dataset)
    data = pd.read_csv('C:/Users/Administrator/Documents/customer/customer_data.csv')
    
    # Initialize the ChurnPredictor
    churn_predictor = ChurnPredictor(data)
    
    # Preprocess the data
    X, y = churn_predictor.preprocess_data()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = churn_predictor.split_data(X, y)
    
    # Train the model
    churn_predictor.train_model(X_train, y_train)
    
    # Predict churn for the test set
    y_pred = churn_predictor.predict_churn(X_test)
    
    # Evaluate the model's performance
    accuracy, precision, recall = churn_predictor.evaluate_performance(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    
    # Calculate churn probabilities
    churn_probabilities = churn_predictor.calculate_churn_probability(X_test)
    print(f"Churn Probabilities: {churn_probabilities}")
    
    # Calculate the customer retention rate
    retention_rate = churn_predictor.calculate_retention_rate()
    print(f"Customer Retention Rate: {retention_rate:.2f}")
