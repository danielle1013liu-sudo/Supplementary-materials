import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def remove_outliers_iqr(df, column, multiplier=1.5):
    """Remove outliers using the IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    print(f"Removed {outliers_mask.sum()} outliers")
    
    return df[~outliers_mask].copy()

def create_advanced_features(X):
    """Create advanced engineered features"""
    X_enhanced = X.copy()
    
    # Numerical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        # Interaction terms
        if 'Cultivation Time' in numeric_cols and 'Degradation time' in numeric_cols:
            X_enhanced['Time_Interaction'] = X_enhanced['Cultivation Time'] * X_enhanced['Degradation time']
            X_enhanced['Time_Ratio'] = X_enhanced['Cultivation Time'] / (X_enhanced['Degradation time'] + 1e-8)
        
        if 'Aperture Size' in numeric_cols and 'Modulus' in numeric_cols:
            X_enhanced['Size_Modulus'] = X_enhanced['Aperture Size'] * X_enhanced['Modulus']
        
        if 'Aperture Size' in numeric_cols and 'Cultivation Time' in numeric_cols:
            X_enhanced['Size_Time'] = X_enhanced['Aperture Size'] * X_enhanced['Cultivation Time']
    
    # Feature transformations
    if 'Aperture Size' in numeric_cols:
        X_enhanced['Aperture_Size_Squared'] = X_enhanced['Aperture Size'] ** 2
        X_enhanced['Aperture_Size_Log'] = np.log1p(X_enhanced['Aperture Size'])
    
    if 'Cultivation Time' in numeric_cols:
        X_enhanced['Cultivation_Time_Sqrt'] = np.sqrt(X_enhanced['Cultivation Time'] + 1e-8)
    
    # Categorical feature combinations
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) >= 2:
        if 'Material Type' in categorical_cols and 'Functionalization' in categorical_cols:
            X_enhanced['Material_Function'] = X_enhanced['Material Type'].astype(str) + '_' + X_enhanced['Functionalization'].astype(str)
        
        if 'Polarization' in categorical_cols and 'Crosslink_Method' in categorical_cols:
            X_enhanced['Polar_Crosslink'] = X_enhanced['Polarization'].astype(str) + '_' + X_enhanced['Crosslink_Method'].astype(str)
    
    return X_enhanced

def safe_mape(y_true, y_pred, eps=1e-8):
    """Compute MAPE (%) with zero-division protection"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def main():
    print("Final IL-10 Model - Polynomial Features + XGBoost")
    
    # Load and preprocess data
    df = pd.read_excel(r"C:\\Users\\Cherry\\Desktop\\PNG\\IL10-3.xlsx")
    df.columns = df.columns.str.strip()
    
    print(f"Raw data shape: {df.shape}")
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df, 'IL-10', multiplier=1.5)
    print(f"Shape after outlier removal: {df_clean.shape}")
    
    X = df_clean.drop(columns=['IL-10'])
    y = df_clean['IL-10']
    
    # Feature engineering
    print("Performing feature engineering...")
    X_enhanced = create_advanced_features(X)
    print(f"Feature engineering: {X.shape[1]} → {X_enhanced.shape[1]} features")
    
    # Identify categorical and numerical features
    categorical_cols = X_enhanced.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_enhanced.select_dtypes(exclude=['object']).columns.tolist()
    
    seeds = [42, 173, 382]
    results = []
    
    for seed in seeds:
        print(f"\nRunning with random seed: {seed}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=seed
        )
        
        # Numerical preprocessing
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        X_train_num_processed = num_transformer.fit_transform(X_train[numerical_cols])
        X_test_num_processed = num_transformer.transform(X_test[numerical_cols])
        
        # Polynomial features (interaction only)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_num_processed)
        X_test_poly = poly.transform(X_test_num_processed)
        
        # Categorical preprocessing
        if len(categorical_cols) > 0:
            cat_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            X_train_cat = cat_transformer.fit_transform(X_train[categorical_cols])
            X_test_cat = cat_transformer.transform(X_test[categorical_cols])
            
            # Combine numerical and categorical
            X_train_final = np.hstack([X_train_poly, X_train_cat])
            X_test_final = np.hstack([X_test_poly, X_test_cat])
        else:
            X_train_final = X_train_poly
            X_test_final = X_test_poly
        
        # XGBoost model (manual hyperparameters)
        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42
        )
        
        model.fit(X_train_final, y_train)
        y_pred_train = model.predict(X_train_final)
        y_pred_test = model.predict(X_test_final)
        
        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mape_test = safe_mape(y_test, y_pred_test)  # New: MAPE
        
        results.append({
            'seed': seed,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'mape_test_%': mape_test
        })
        
        print(f"Train R²: {r2_train:.4f}")
        print(f"Test R²: {r2_test:.4f}")
        print(f"Test MSE: {mse_test:.2f}")
        print(f"Test MAE: {mae_test:.2f}")
        print(f"Test MAPE (%): {mape_test:.2f}")
    
    # Aggregate results
    results_df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print("Final IL-10 Model Results")
    print(f"{'='*50}")
    print(f"Average Test R²: {results_df['r2_test'].mean():.4f} ± {results_df['r2_test'].std():.4f}")
    print(f"Average Test MSE: {results_df['mse_test'].mean():.2f} ± {results_df['mse_test'].std():.2f}")
    print(f"Average Test MAE: {results_df['mae_test'].mean():.2f} ± {results_df['mae_test'].std():.2f}")
    print(f"Average Test MAPE (%): {results_df['mape_test_%'].mean():.2f} ± {results_df['mape_test_%'].std():.2f}")
    
    # Save results
    results_df.to_csv('il10_final_results.csv', index=False)
    print("\nResults saved to il10_final_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = main()
