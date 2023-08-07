import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the dataset from Google Drive
# file_path = '/content/drive/My Drive/ato_dataset.csv'
data = pd.read_csv('ato_dataset.csv')

# Convert 'Timestamp' to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Preprocess the data (e.g., label encoding or other preprocessing)
# ...

# One-hot encode categorical features
categorical_columns = ['Usual Device', 'Current Device', 'Usual IP', 'Current IP']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split the data into features (X) and target (y)
X = data_encoded.drop(['Timestamp', 'Is ATO'], axis=1)
y = data_encoded['Is ATO']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initialize an XGBoostClassifier
clf = XGBClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


pickle.dump(clf,open('model.pkl','wb'))

modell=pickle.load(open('model.pkl','rb'))










# # Importing essential libraries
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# import pickle


# # Loading the dataset
# df = pd.read_csv('Admission_Predict.csv')

# # Renaming the columns with appropriate names
# df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'Probability'})
# df.head()

# # Data Cleaning

# # Removing the serial no, column
# df.drop('Serial No.', axis='columns', inplace=True)  # setting inplace =True then it fills values at an empty place.
# df.head()

# # Replacing the 0 values from ['GRE','TOEFL','University Rating','SOP','LOR','CGPA'] by NaN
# df_copy = df.copy(deep=True)   # data is copied but actual Python objects will not be copied recursively
# df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']] = df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']].replace(0, np.NaN)

# #Checking if there is null value in the dataset or not
# df_copy.isnull().sum()


# # Model Building

# # Splitting the dataset in features and label
# X = df_copy.drop('Probability', axis='columns')
# y = df_copy['Probability']


# # Using GridSearchCV to find the best algorithm for this problem
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor


# # Creating a function to calculate best model for this problem
# def find_best_model(X, y):
#     models = {
#         'linear_regression': {
#             'model': LinearRegression(),
#             'parameters': {
#                 'normalize': [True,False]
#             }
#         },
         
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'parameters': {
#                 'criterion': ['mse', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         },
        
#         'knn': {
#             'model': KNeighborsRegressor(algorithm='auto'),
#             'parameters': {
#                 'n_neighbors': [2,5,10,20]
#             }
#         }
#     }
    
#     scores = []
#     for model_name, model_params in models.items():
#         gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False) # Cross validation cv=5 (n-fold cv)
#         gs.fit(X, y)
#         scores.append({
#             'model': model_name,
#             'best_parameters': gs.best_params_,
#             'score': gs.best_score_
#         })
        
#     return pd.DataFrame(scores, columns=['model','best_parameters','score'])
        
# find_best_model(X, y)


# '''

#         model	             best_parameters	                               score
# 1	    linear_regression   {'normalize': True}	                              0.810802
# 2	    decision_tree	    {'criterion': 'mse', 'splitter': 'random'}	      0.586808
# 3	    knn	                {'n_neighbors': 20}	                              0.722961


# '''



# # Using cross_val_score for gaining highest accuracy
# # calculating accuracy seprating for linear regression
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(LinearRegression(normalize=True), X, y, cv=5)
# print('Highest Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))


# # Splitting the dataset into train and test samples
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
# print(len(X_train), len(X_test))
# # Output: 400 100

# # Creating Linear Regression Model
# model = LinearRegression(normalize=True)
# model.fit(X_train, y_train)
# model.score(X_test, y_test)



# # Prediction 1
# # Input in the form : GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
# print('Chance of getting into UCLA is {}%'.format(round(model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])[0]*100, 3)))


# # Prediction 2
# # Input in the form : GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
# print('Chance of getting into UCLA is {}%'.format(round(model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])[0]*100, 3)))

# pickle.dump(model,open('model.pkl','wb'))

# modell=pickle.load(open('model.pkl','rb'))

# # print(modell.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]]))
# # Total Scores of Exams
# # GRE- 340
# #TOEFL=120
# #LOR- 5
# #SOp-5

