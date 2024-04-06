import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/hitters.csv')

###########Data Cleaning#########
# Check for missing values
print(df.isnull().sum())

# Remove or impute missing values
df = df.dropna()  # or use an imputer

################Feature Selection and Engineering:#######################
# Select relevant features, assuming 'Salary' is the target variable
features = df.drop('Salary', axis=1)
target = df['Salary']

# Feature engineering, for example, create a 'Batting Average' feature
features['BattingAverage'] = features['Hits'] / features['AtBat']

####################Preprocessing:#####################
# Encode categorical variables
categorical_features = ['League', 'Division', 'NewLeague']
numeric_features = features.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer to transform features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

##############Model Selection:####################
# Initialize the model
model = RandomForestRegressor(n_estimators=100)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

##################Splitting the Data:########################
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

############Training the Model:##################
# Fit the model
pipeline.fit(X_train, y_train)

##############Cross-Validation:#############
# Perform k-fold cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f'CV average score: {np.mean(cv_scores)}')

################Model Evaluation:################
# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse}')

######################Visualization (optional):###################
# Plot actual vs predicted salaries
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salaries')
plt.ylabel('Predicted Salaries')
plt.title('Actual vs Predicted Salaries')
plt.show()
