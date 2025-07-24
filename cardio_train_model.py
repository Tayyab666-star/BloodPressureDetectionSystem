import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Import joblib for saving/loading models

print("--- Starting Model Training and Saving Script ---")

# --- 1. Load the Dataset ---
try:
    # The dataset typically comes with an 'id' column which is not useful for training
    df = pd.read_csv('cardio_train.csv', sep=';') # Note: This dataset uses semicolon as separator
    df = df.drop('id', axis=1) # Drop the 'id' column
    print("Dataset 'cardio_train.csv' loaded successfully!")
except FileNotFoundError:
    print("Error: 'cardio_train.csv' not found. Please make sure the file is in the same directory.")
    print("You can download it from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
    exit()

print("\nOriginal Data Head:")
print(df.head())
print("\nData Info:")
df.info()

# --- 2. Data Preprocessing ---

# A. Convert 'age' from days to years
df['age_years'] = (df['age'] / 365).astype(int)
df = df.drop('age', axis=1) # Drop original 'age' in days
print("\n'age' column converted to 'age_years'.")

# B. Rename blood pressure columns for clarity
df = df.rename(columns={'ap_hi': 'SystolicBP', 'ap_lo': 'DiastolicBP'})
print("Blood pressure columns renamed to 'SystolicBP' and 'DiastolicBP'.")

# C. Handle potential outliers/erroneous blood pressure readings
initial_rows = len(df)
df = df[(df['SystolicBP'] > 70) & (df['SystolicBP'] < 200)]
df = df[(df['DiastolicBP'] > 40) & (df['DiastolicBP'] < 140)]
df = df[df['SystolicBP'] >= df['DiastolicBP']] # Systolic must be >= Diastolic
print(f"Dropped {initial_rows - len(df)} rows due to implausible blood pressure values.")

# D. Simulate Computer Vision Features
num_samples = len(df)
np.random.seed(42) # for reproducibility

df['FacialRednessIndex'] = np.random.uniform(0.1, 0.9, num_samples)
df['EyeAreaRatio'] = np.random.uniform(0.01, 0.05, num_samples)
df['SkinToneVariability'] = np.random.uniform(0.001, 0.01, num_samples)
df['EstimatedHeartRate_CV'] = np.random.randint(60, 100, num_samples)
df['PPG_SignalNoiseRatio'] = np.random.uniform(5, 30, num_samples)
print("Simulated Computer Vision Features added.")

print("\nData Head with CV Features (and age in years, renamed BP):")
print(df.head())


# E. Define Features (X) and Target Variables (y)
X = df.drop(['SystolicBP', 'DiastolicBP', 'cardio'], axis=1)
y_systolic = df['SystolicBP']
y_diastolic = df['DiastolicBP']
print("\nFeatures (X) and Target Variables (y) defined.")

# F. Identify Numerical and Categorical Features
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
    if col in numerical_cols:
        numerical_cols.remove(col)
    if col not in categorical_cols: # Ensure it's added if not already
        categorical_cols.append(col)

print(f"\nIdentified Numerical columns: {numerical_cols}")
print(f"Identified Categorical columns: {categorical_cols}")

# G. Handle Missing Values and Encode Categorical Features using ColumnTransformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)
print("Preprocessor (imputation, scaling, one-hot encoding) defined.")


# H. Split the data into training and testing sets
X_train_sys, X_test_sys, y_train_sys, y_test_sys = train_test_split(
    X, y_systolic, test_size=0.2, random_state=42
)

X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(
    X, y_diastolic, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# Apply the preprocessor (imputation, scaling, and one-hot encoding)
X_train_sys_processed = preprocessor.fit_transform(X_train_sys)
X_test_sys_processed = preprocessor.transform(X_test_sys)

# Use the same fitted preprocessor for diastolic as features are identical
X_train_dia_processed = preprocessor.transform(X_train_dia)
X_test_dia_processed = preprocessor.transform(X_test_dia)

print(f"\nProcessed Training data shape for Systolic BP: {X_train_sys_processed.shape}")
print(f"Processed Testing data shape for Systolic BP: {X_test_sys_processed.shape}")

# --- 3. Train the Models ---

# Initialize the Random Forest Regressor for Systolic BP
rf_systolic_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining Systolic BP model...")
rf_systolic_model.fit(X_train_sys_processed, y_train_sys)
print("Systolic BP model training complete.")

# Initialize the Random Forest Regressor for Diastolic BP
rf_diastolic_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("Training Diastolic BP model...")
rf_diastolic_model.fit(X_train_dia_processed, y_train_dia)
print("Diastolic BP model training complete.")

# --- 4. Evaluate the Models ---

# Make predictions and evaluate for Systolic BP
y_pred_sys = rf_systolic_model.predict(X_test_sys_processed)
mae_sys = mean_absolute_error(y_test_sys, y_pred_sys)
mse_sys = mean_squared_error(y_test_sys, y_pred_sys)
rmse_sys = np.sqrt(mse_sys)
r2_sys = r2_score(y_test_sys, y_pred_sys)

print(f"\n--- Systolic Blood Pressure Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_sys:.2f}")
print(f"Mean Squared Error (MSE): {mse_sys:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_sys:.2f}")
print(f"R-squared (R²): {r2_sys:.2f}")

# Make predictions and evaluate for Diastolic BP
y_pred_dia = rf_diastolic_model.predict(X_test_dia_processed)
mae_dia = mean_absolute_error(y_test_dia, y_pred_dia)
mse_dia = mean_squared_error(y_test_dia, y_pred_dia)
rmse_dia = np.sqrt(mse_dia)
r2_dia = r2_score(y_test_dia, y_pred_dia)

print(f"\n--- Diastolic Blood Pressure Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae_dia:.2f}")
print(f"Mean Squared Error (MSE): {mse_dia:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_dia:.2f}")
print(f"R-squared (R²): {r2_dia:.2f}")

# --- Optional: Visualize Predictions vs Actuals ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.regplot(x=y_test_sys, y=y_pred_sys, scatter_kws={'alpha':0.3})
plt.xlabel("Actual Systolic BP")
plt.ylabel("Predicted Systolic BP")
plt.title("Systolic Blood Pressure: Actual vs. Predicted")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.regplot(x=y_test_dia, y=y_pred_dia, scatter_kws={'alpha':0.3})
plt.xlabel("Actual Diastolic BP")
plt.ylabel("Predicted Diastolic BP")
plt.title("Diastolic Blood Pressure: Actual vs. Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: Feature Importance ---
try:
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(ohe_feature_names)
except KeyError as e:
    print(f"\nWarning: Could not retrieve feature names from preprocessor. Error: {e}")
    all_feature_names = [f"feature_{i}" for i in range(X_train_sys_processed.shape[1])]

if rf_systolic_model.feature_importances_.shape[0] == len(all_feature_names):
    feature_importances_sys = pd.Series(rf_systolic_model.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    feature_importances_dia = pd.Series(rf_diastolic_model.feature_importances_, index=all_feature_names).sort_values(ascending=False)

    print("\n--- Systolic BP Feature Importances (Top 10) ---")
    print(feature_importances_sys.head(10))

    print("\n--- Diastolic BP Feature Importances (Top 10) ---")
    print(feature_importances_dia.head(10))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=feature_importances_sys.head(10), y=feature_importances_sys.head(10).index)
    plt.title("Top 10 Systolic BP Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.subplot(1, 2, 2)
    sns.barplot(x=feature_importances_dia.head(10), y=feature_importances_dia.head(10).index)
    plt.title("Top 10 Diastolic BP Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping Feature Importance visualization due to mismatch in feature count.")
    print(f"Model feature importances count: {rf_systolic_model.feature_importances_.shape[0]}")
    print(f"Constructed feature names count: {len(all_feature_names)}")


# --- How to use the trained model for new predictions ---
def predict_blood_pressure_new_data(model_sys, model_dia, preprocessor_obj, new_data_dict, feature_columns):
    """
    Predicts systolic and diastolic blood pressure for new input data, including CV features.
    """
    new_data_df = pd.DataFrame([new_data_dict], columns=feature_columns)
    new_data_processed = preprocessor_obj.transform(new_data_df)

    predicted_systolic = model_sys.predict(new_data_processed)[0]
    predicted_diastolic = model_dia.predict(new_data_processed)[0]

    print(f"\n--- Prediction for New Data ---")
    print(f"Input: {new_data_dict}")
    print(f"Predicted Systolic BP: {predicted_systolic:.2f} mmHg")
    print(f"Predicted Diastolic BP: {predicted_diastolic:.2f} mmHg")

# Example of making a new prediction
example_features_for_prediction = {
    'gender': 1, # Male
    'height': 170, # cm
    'weight': 75, # kg
    'cholesterol': 1, # Normal
    'gluc': 1, # Normal
    'smoke': 0, # No
    'alco': 0, # No
    'active': 1, # Yes
    'age_years': 45, # in years
    'FacialRednessIndex': 0.65, # Simulated CV feature
    'EyeAreaRatio': 0.03,        # Simulated CV feature
    'SkinToneVariability': 0.005, # Simulated CV feature
    'EstimatedHeartRate_CV': 72, # Simulated CV feature
    'PPG_SignalNoiseRatio': 25.0 # Simulated CV feature
}

original_feature_columns_for_prediction = X.columns.tolist()

predict_blood_pressure_new_data(rf_systolic_model, rf_diastolic_model, preprocessor,
                                example_features_for_prediction, original_feature_columns_for_prediction)

# Another example
example_features_for_prediction_2 = {
    'gender': 2, # Female
    'height': 160,
    'weight': 90,
    'cholesterol': 3, # High
    'gluc': 2, # High
    'smoke': 1, # Yes
    'alco': 1, # Yes
    'active': 0, # No
    'age_years': 60,
    'FacialRednessIndex': 0.8,
    'EyeAreaRatio': 0.02,
    'SkinToneVariability': 0.008,
    'EstimatedHeartRate_CV': 90,
    'PPG_SignalNoiseRatio': 10.0
}
predict_blood_pressure_new_data(rf_systolic_model, rf_diastolic_model, preprocessor,
                                example_features_for_prediction_2, original_feature_columns_for_prediction)


# --- SAVE THE TRAINED MODELS AND PREPROCESSOR ---
print("\nSaving trained models and preprocessor...")
joblib.dump(rf_systolic_model, 'systolic_bp_model.pkl')
joblib.dump(rf_diastolic_model, 'diastolic_bp_model.pkl')
joblib.dump(preprocessor, 'data_preprocessor.pkl')
print("Models and preprocessor saved successfully!")
print("--- Model Training and Saving Script Finished ---")
