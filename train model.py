import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and preprocess the dataset
train_data = pd.read_excel(r'C:\Users\suhas\OneDrive\Documents\Crime_Prediction\data\Train.xlsx')
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data['YEAR'] = train_data['Date'].dt.year
train_data['MONTH'] = train_data['Date'].dt.month

# Encode the 'type' column
label_encoder = LabelEncoder()
train_data['type_encoded'] = label_encoder.fit_transform(train_data['TYPE'])

# Aggregate the data by year, month, and type
monthly_crime_data = train_data.groupby(['YEAR', 'MONTH', 'type_encoded']).size().reset_index(name='crime_count')
X = monthly_crime_data[['YEAR', 'MONTH', 'type_encoded']]
y = monthly_crime_data['crime_count']

# Train the model with verbose output
tscv = TimeSeriesSplit(n_splits=3)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

# Iterate through each train/test split and fit the model
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print("Training on split:")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Save the model to a file
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
