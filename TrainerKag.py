import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. LOAD DATA
train_path = r'C:\Users\br300\Documents\VScodeprojectsprof\MLDatasets\train.csv'
test_path = r'C:\Users\br300\Documents\VScodeprojectsprof\MLDatasets\test.csv'

data = pd.read_csv(train_path)
testd = pd.read_csv(test_path)

# 2. FEATURE ENGINEERING (The Human Architect Part)
diff_map = {'Low': 1, 'Medium': 2, 'High': 3} 

def engineer_features(df):
    temp_df = df.copy()
    # Logic: Map difficulty to numbers for math
    temp_df['diff_num'] = temp_df['exam_difficulty'].map(diff_map).fillna(2)
    # Logic: Interaction between difficulty and study hours
    temp_df['difstud'] = temp_df['diff_num'] / temp_df['study_hours']
    temp_df['difstud'] = temp_df['difstud'].replace([np.inf, -np.inf], 0).fillna(0)
    return temp_df

data = engineer_features(data)
testd = engineer_features(testd)

cols = ["course", "class_attendance", "internet_access", "sleep_hours", 
        "sleep_quality", "study_method", "difstud", "facility_rating"]

X = data[cols]
y = data['exam_score']

# 3. PREPROCESSING PIPELINE
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=['object']))
    ])

# 4. PREPARE NATIVE DMATRIX
# We transform the data first so it's ready for Native XGBoost CV
X_prepped = preprocessor.fit_transform(X)
dtrain = xgb.DMatrix(X_prepped, label=y)

# 5. CROSS-VALIDATION SETTINGS
# Using your requested hyperparams
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 12,
    'eval_metric': 'rmse',
    'seed': 42
}

print("Starting 5-Fold Cross-Validation...")

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=5,
    verbose_eval=False,
    as_pandas=True,
    seed=42
)

# 6. ANALYZE CV RESULTS
best_n = cv_results.shape[0]
final_train_rmse = cv_results['train-rmse-mean'].iloc[-1]
final_val_rmse = cv_results['test-rmse-mean'].iloc[-1]
gap = final_val_rmse - final_train_rmse

print(f"\n--- Cross-Validation Results ---")
print(f"Optimal Trees (n_estimators): {best_n}")
print(f"Average Train RMSE: {final_train_rmse:.4f}")
print(f"Average Val RMSE (The 'Truth'): {final_val_rmse:.4f}")
print(f"Average RMSE Gap: {gap:.4f}")

# 7. TRAIN FINAL MODEL ON 100% OF DATA
# Now we use the 'best_n' we found across all folds
final_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_n
)

# 8. PREDICT ON TEST DATA
X_test_prepped = preprocessor.transform(testd[cols])
dtest = xgb.DMatrix(X_test_prepped)
y_test_pred = final_model.predict(dtest)

# 9. EXPORT
output = pd.DataFrame({'Id': testd.id, 'exam_score': y_test_pred})
save_path = r'C:\Users\br300\Documents\VScodeprojectsprof\MLProjs\submission.csv'
output.to_csv(save_path, index=False)

print(f"\nFinal submission saved with {best_n} trees.")
