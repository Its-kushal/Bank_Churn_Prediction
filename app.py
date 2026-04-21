from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('churn.csv')

# Preprocessing
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)
X = data.drop('Exited', axis=1)
y = data['Exited']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))

# Train K Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))

# Train Support Vector Machine classifier
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))

# Train Logistic Regression classifier
logreg_classifier = LogisticRegression(random_state=42)
logreg_classifier.fit(X_train, y_train)
logreg_accuracy = accuracy_score(y_test, logreg_classifier.predict(X_test))

# Determine the classifier with the highest accuracy
classifiers = {'Random Forest': rf_accuracy, 'K Nearest Neighbors': knn_accuracy,
               'Support Vector Machine': svm_accuracy, 'Logistic Regression': logreg_accuracy}
best_classifier = max(classifiers, key=classifiers.get)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        credit_score = float(request.form['credit_score'])
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        estimated_salary = float(request.form['estimated_salary'])
        has_credit_card = int(request.form.get('has_credit_card', 0))
        is_active_member = int(request.form.get('is_active_member', 0))
        
        # Prepare input data for prediction
        input_data = np.array([[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, 0, 0, 1]])
        
        # Use the best classifier for prediction
        if best_classifier == 'Random Forest':
            churn_prediction = rf_classifier.predict(scaler.transform(input_data))
        elif best_classifier == 'K Nearest Neighbors':
            churn_prediction = knn_classifier.predict(scaler.transform(input_data))
        elif best_classifier == 'Support Vector Machine':
            churn_prediction = svm_classifier.predict(scaler.transform(input_data))
        elif best_classifier == 'Logistic Regression':
            churn_prediction = logreg_classifier.predict(scaler.transform(input_data))
        
        offers = generate_retention_offers(input_data)
        print("Generated offers:", offers)  # Debug print statement
        return render_template('result.html', churn_prediction=churn_prediction[0], offers=offers)
    
    except Exception as e:
        return render_template('result.html', error_message=str(e))

def generate_retention_offers(input_data):
    offers = []
    credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, _, _, _ = input_data[0]

    # Offer based on credit score
    if credit_score < 600:
        offers.append('Offering credit score improvement program.')
    elif credit_score > 800:
        offers.append('Providing exclusive VIP banking services.')

    # Offer based on age
    if age > 60:
        offers.append('Special retirement planning services.')

    # Offer based on tenure and balance
    if tenure > 5 and balance > 50000:
        offers.append('Exclusive benefits for long-time customers with high balance.')

    # Offer based on number of products
    if num_of_products > 2:
        offers.append('Upgrade to premium account for free.')

    # Offer based on estimated salary
    if estimated_salary > 100000:
        offers.append('Personalized investment opportunities.')

    # Common offers
    if has_credit_card == 1:
        offers.append('Upgrade credit card with better rewards.')
    if is_active_member == 1:
        offers.append('Loyalty rewards for staying active.')

    return offers


if __name__ == '__main__':
    app.run(debug=True)
