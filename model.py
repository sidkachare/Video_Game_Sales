import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv('vgsales.csv')

# Drop missing values from the dataset
data = data.dropna(subset=['Year', 'Publisher'], axis=0).reset_index(drop=True)

# Prepare the target variable
y = data['Global_Sales']

# Prepare the input features
features = data[['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
categorical_cols = ['Platform', 'Genre', 'Publisher']

# Convert features to a DataFrame
features_df = pd.DataFrame(features)

# Create the pipeline for one-hot encoding and conversion to numeric format
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=0)

# Train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
model.fit(X_train, y_train)

# Save the model as a pickle file
pickle.dump(model, open("model.pkl", "wb"))
