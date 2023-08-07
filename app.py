from flask import Flask, render_template, request, jsonify
import csv
import pandas as pd

app = Flask(__name__)


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index.html')

@app.route('/manualInputs')
def manualInputs():
    return render_template('manualInputs.html')

@app.route('/uploadCSV')
def uploadCSV():
    return render_template('uploadCSV.html')

def get_prediction(failedLogins):
    prediction=''
    if(failedLogins>=5):
        prediction='ATO'
    else:
        prediction='Non-ATO'
    
    return prediction

def append_extra_column(filename, list_to_append):
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list_to_append)

def get_extra_row(filename):
    extraRow = []
    extraRow.append("prediction")
    rowData = pd.read_csv(filename)
    failedLogins = rowData['Failed Logins']
    for item in failedLogins:
        extraRow.append(get_prediction(item))
    return extraRow

@app.route('/process_csv', methods=['POST'])
def process_csv():
    file = request.files['csv_file']
    if file and file.filename.endswith('.csv'):
        csv_data = file.read().decode('utf-8')
        csv_values = csv.reader(csv_data.splitlines())
        result = process_data(csv_values)
        return jsonify(result)

def process_data(data):
    processed_data = []
    for row in data:
        processed_data.append({'column1': row[0], 'column2': row[1], 'column3': row[2], 'column4': row[3], 'column5': row[4], 'column6': row[5], 'column7': row[6], 'column8': row[7],'column9': row[8]})
    return processed_data



from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder



# Load and preprocess the dataset
def load_and_preprocess_data():
    file_path = 'ato_dataset.csv'  # Update with your dataset path
    data = pd.read_csv(file_path)

    # Preprocess the data as needed (fill in missing values, etc.)
    # ...

    # One-hot encode categorical features
    categorical_columns = ['Usual Device', 'Current Device', 'Usual IP', 'Current IP']
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Remove Timestamp column
    data_encoded = data_encoded.drop(['Timestamp'], axis=1)

    # Split the data into features (X) and target (y)
    X = data_encoded.drop(['Is ATO'], axis=1)
    y = data_encoded['Is ATO']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist()

# Train the model
def train_model(X_train, y_train):
    clf = XGBClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Calculate accuracy on test set
def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        X_train, X_test, y_train, y_test, columns = load_and_preprocess_data()
        clf = train_model(X_train, y_train)

        session_duration = float(request.form['session_duration'])
        failed_logins = int(request.form['failed_logins'])
        usual_location = int(request.form['usual_location'])
        time_difference = float(request.form['time_difference'])
        usual_device = str(request.form['usual_device'])
        current_device = str(request.form['current_device'])
        usual_ip =str( request.form['usual_ip'])
        current_ip = str(request.form['current_ip'])

        # Create a DataFrame with the input data for one-hot encoding
        input_data = pd.DataFrame({
            'Session Duration (s)': [session_duration],
            'Failed Logins': [failed_logins],
            'Usual Location': [usual_location],
            'Time Difference (mins)': [time_difference],
            'Usual Device': [usual_device],
            'Current Device': [current_device],
            'Usual IP': [usual_ip],
            'Current IP': [current_ip]
        })

        # Apply one-hot encoding to the categorical variables
        input_data_encoded = pd.get_dummies(input_data, columns=['Usual Device', 'Current Device', 'Usual IP', 'Current IP'], drop_first=True)

        # Reindex the input_data_encoded DataFrame to match the columns used during training
        input_data_encoded = input_data_encoded.reindex(columns=columns, fill_value=0)

        # Make the prediction
        prediction = clf.predict(input_data_encoded)


        prediction_label = 'ATO' if failed_logins <= 5 else 'Non-ATO'

        # Calculate accuracy (this step may not be necessary here)
        accuracy = calculate_accuracy(clf, X_test, y_test)

        return render_template('manualInputs.html', prediction=prediction_label, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True,port=5005)
