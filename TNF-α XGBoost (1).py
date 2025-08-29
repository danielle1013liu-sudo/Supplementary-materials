import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(X):
    X_enhanced = X.copy()
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        if 'Cultivation Time' in numeric_cols and 'Degradation time' in numeric_cols:
            X_enhanced['Time_Interaction'] = X_enhanced['Cultivation Time'] * X_enhanced['Degradation time']
            X_enhanced['Time_Ratio'] = X_enhanced['Cultivation Time'] / (X_enhanced['Degradation time'] + 1e-8)
        
        if 'Cultivation Time' in numeric_cols and 'Aperture Size' in numeric_cols:
            X_enhanced['Cultivation_Aperture'] = X_enhanced['Cultivation Time'] * X_enhanced['Aperture Size']
        
        if 'Degradation time' in numeric_cols and 'Modulus' in numeric_cols:
            X_enhanced['Degradation_Modulus'] = X_enhanced['Degradation time'] * X_enhanced['Modulus']
    
    if 'Cultivation Time' in numeric_cols:
        X_enhanced['Cultivation_Time_Squared'] = X_enhanced['Cultivation Time'] ** 2
        X_enhanced['Cultivation_Time_Log'] = np.log1p(X_enhanced['Cultivation Time'])
    
    if 'Degradation time' in numeric_cols:
        X_enhanced['Degradation_Time_Sqrt'] = np.sqrt(X_enhanced['Degradation time'] + 1e-8)
    
    if 'Aperture Size' in numeric_cols:
        X_enhanced['Aperture_Size_Log'] = np.log1p(X_enhanced['Aperture Size'])
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) >= 2:
        if 'Material Type' in categorical_cols and 'Macrophage_Type' in categorical_cols:
            X_enhanced['Material_Macrophage'] = X_enhanced['Material Type'].astype(str) + '_' + X_enhanced['Macrophage_Type'].astype(str)
        
        if 'Functionalization' in categorical_cols and 'Crosslink_Method' in categorical_cols:
            X_enhanced['Function_Crosslink'] = X_enhanced['Functionalization'].astype(str) + '_' + X_enhanced['Crosslink_Method'].astype(str)
    
    return X_enhanced

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def main():
    print("TNF-α XGBoost Model")
    
    df = pd.read_excel(r"C:\\Users\\Cherry\\Desktop\\PNG\\TNFA3.xlsx")
    df.columns = df.columns.str.strip()
    
    print(f"Raw dataset shape: {df.shape}")
    
    X = df.drop(columns=['TNF-α'])
    y = np.log1p(df['TNF-α']) 
    
    print(f"Log-transformed target range: {y.min():.2f} - {y.max():.2f}")
    
    print("Performing feature engineering...")
    X_enhanced = create_advanced_features(X)
    print(f"Feature engineering: {X.shape[1]} → {X_enhanced.shape[1]} features")
    
    categorical_cols = X_enhanced.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_enhanced.select_dtypes(exclude=['object']).columns.tolist()
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    seeds = [42, 793, 235]
    results = []
    
    for seed in seeds:
        print(f"\nRunning with random seed: {seed}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=seed
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        model = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_processed, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred_train_log = best_model.predict(X_train_processed)
        y_pred_test_log = best_model.predict(X_test_processed)
        
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        y_pred_train_orig = np.expm1(y_pred_train_log)
        y_pred_test_orig = np.expm1(y_pred_test_log)
        
        r2_train = r2_score(y_train_orig, y_pred_train_orig)
        r2_test = r2_score(y_test_orig, y_pred_test_orig)
        mse_test = mean_squared_error(y_test_orig, y_pred_test_orig)
        mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)
        mape_test = safe_mape(y_test_orig, y_pred_test_orig)
        
        r2_test_log = r2_score(y_test, y_pred_test_log)
        
        results.append({
            'seed': seed,
            'best_params': str(grid_search.best_params_),
            'r2_train_orig': r2_train,
            'r2_test_orig': r2_test,
            'r2_test_log': r2_test_log,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'mape_test_%': mape_test
        })
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test R² (log scale): {r2_test_log:.4f}")
        print(f"Train R² (original scale): {r2_train:.4f}")
        print(f"Test R² (original scale): {r2_test:.4f}")
        print(f"Test MSE: {mse_test:.2f}")
        print(f"Test MAE: {mae_test:.2f}")
        print(f"Test MAPE (%): {mape_test:.2f}")
    
    results_df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print("Final TNF-α Model Results")
    print(f"{'='*50}")
    print(f"Average Test R² (original scale): {results_df['r2_test_orig'].mean():.4f} ± {results_df['r2_test_orig'].std():.4f}")
    print(f"Average Test R² (log scale): {results_df['r2_test_log'].mean():.4f} ± {results_df['r2_test_log'].std():.4f}")
    print(f"Average Test MSE: {results_df['mse_test'].mean():.2f} ± {results_df['mse_test'].std():.2f}")
    print(f"Average Test MAE: {results_df['mae_test'].mean():.2f} ± {results_df['mae_test'].std():.2f}")
    print(f"Average Test MAPE (%): {results_df['mape_test_%'].mean():.2f} ± {results_df['mape_test_%'].std():.2f}")
    
    results_df.to_csv('tnfa_final_results.csv', index=False)
    print("\nResults saved to tnfa_final_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = main()
