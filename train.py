import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import pickle

# Load the dataset
data = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\99acres\house\Book1.csv")

# Preprocessing
data.dropna(inplace=True)
data['size'] = data['size'].astype(str)
data['total_sqft'] = data['total_sqft'].astype(str)
data['size'] = data['size'].str.extract('(\d+)').astype(float)
data['total_sqft'] = data['total_sqft'].str.extract('(\d+\.?\d*)').astype(float)

# Split the data into features and target variable
X = data[['total_sqft', 'bath', 'location', 'size']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical features using KNN imputation
numeric_features = ['total_sqft', 'bath', 'size']
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler())])

# Preprocessing for categorical features
categorical_features = ['location']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to a file
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Example prediction for a new data point
location="ISRO Layout"
sqft=1500
bath=2
bhk=3
new_data_point = pd.DataFrame({'total_sqft': [sqft],
                               'bath': [bath],
                               'location': [location],
                               'size': [bhk]})
with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

predicted_price = loaded_model.predict(new_data_point)
totalIncome = predicted_price * sqft

# Function to calculate years to reach a target value
def years_to_reach_value(principal, target_value, annual_interest_rate):
    years = np.log(target_value / principal) / np.log(1 + annual_interest_rate)
    return years.item()

# Calculate years needed to break even
principal = totalIncome
target_value = prediction
annual_interest_rate = 0.05  # 5% increase every year
years_needed = years_to_reach_value(principal, target_value, annual_interest_rate)

# Print the number of years needed to break even with formatting
print(f"Number of years required to breakeven: {years_needed:.2f}")
