# Random Forest Regressor

# importing modules and packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# importing data
df = pd.read_csv('River Data Sample.csv')
print("River Data Sample.csv")
print("\nCOLUMNS")
print(df.columns)

# printing data
df = df.drop(columns = ['No', ' river', ' stream', ' direction'])
print("\nDATA USED FOR MODELLING")
print(df.iloc[:6])

# creating feature variables
X = df.drop('Y AQI', axis = 1)
y = df['Y AQI']

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, shuffle = False)

# standardizing dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# initialize Random Forest Regressor
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train_scaled,y_train)
y_pred = rfr.predict(X_test_scaled)
print()
print("None", "PREDICTIONS")
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# results + 0-layer feature selection
results = []
results.append(("RFR", "None", "None", mse, mae))


# 1-layer feature selection
feature_selectors = {
    "PCA1": PCA(n_components=1),
    "PCA2": PCA(n_components=2),
    "PCA3": PCA(n_components=3),
    "RFE1": RFE(rfr, n_features_to_select=1),
    "RFE2": RFE(rfr, n_features_to_select=2),
    "RFE3": RFE(rfr, n_features_to_select=3),
    "UNI1": SelectKBest(score_func=f_regression, k=1),
    "UNI2": SelectKBest(score_func=f_regression, k=2),
    "UNI3": SelectKBest(score_func=f_regression, k=3)
}

for (fs_name, fs) in feature_selectors.items():
    
    # Apply feature selection
    X_train_fs = fs.fit_transform(X_train_scaled, y_train)
    X_test_fs = fs.transform(X_test_scaled)
    
    # Train model
    rfr.fit(X_train_fs, y_train)
    
    # Predictions
    y_pred = rfr.predict(X_test_fs)
    print()
    print(fs_name, "PREDICTIONS")
    print(y_pred)
    
    # Compute MSE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append(("RFR", fs_name, "None", mse, mae))


# 2-layer feature selection
feature_selectors1 = {
    "PCA2": PCA(n_components=2),
    "PCA3": PCA(n_components=3),
    "RFE2": RFE(rfr, n_features_to_select=2),
    "RFE3": RFE(rfr, n_features_to_select=3),
    "UNI2": SelectKBest(score_func=f_regression, k=2),
    "UNI3": SelectKBest(score_func=f_regression, k=3)
}

feature_selectors2 = {
    "PCA1": PCA(n_components=1),
    "PCA2": PCA(n_components=2),
    "RFE1": RFE(rfr, n_features_to_select=1),
    "RFE2": RFE(rfr, n_features_to_select=2),
    "UNI1": SelectKBest(score_func=f_regression, k=1),
    "UNI2": SelectKBest(score_func=f_regression, k=2),
}

for (fs1_name, fs1) in feature_selectors1.items():

    for(fs2_name, fs2) in feature_selectors2.items(): 

        if (fs1_name == "PCA2" or fs1_name == "RFE2" or fs1_name == "UNI2") and (fs2_name == "PCA2" or fs2_name == "RFE2" or fs2_name == "UNI2"):
            continue
        
        # Apply 1st feature selection
        X_train_fs1 = fs1.fit_transform(X_train_scaled, y_train)
        X_test_fs1 = fs1.transform(X_test_scaled)

        # Apply 2nd feature selection
        X_train_fs2 = fs2.fit_transform(X_train_fs1, y_train)
        X_test_fs2 = fs2.transform(X_test_fs1)
        
        # Train model
        rfr.fit(X_train_fs2, y_train)
        
        # Predictions
        y_pred = rfr.predict(X_test_fs2)
        print()
        print(fs1_name, fs2_name, "PREDICTIONS")
        print(y_pred)
        
        # Compute MSE
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append(("RFR", fs1_name, fs2_name, mse, mae))

print("\nACTUAL VALUE")
print(y_test)

# Print results
print()
df_results = pd.DataFrame(results, columns=["Model", "Feature_Selection 1", "Feature_Selection 2", "MSE", "MAE"])
print(df_results)
