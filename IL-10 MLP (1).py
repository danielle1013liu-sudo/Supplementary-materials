# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor

# ========================
# Configurations
# ========================
SEEDS = [79, 955]
Y_TRANSFORM = 'quantile'   # 'quantile' or 'yeo-johnson'
APPLY_WINSORIZE_Y = True
WINSOR_HIGH_Q = 0.995
DROP_INTERACTIONS = True

INTERACTION_PREFIXES = [
    'Time_Interaction', 'Time_Ratio',
    'Size_Modulus', 'Size_Time',
    'Material_Function', 'Polar_Crosslink',
]

# ========================
# Feature Engineering
# ========================
def create_advanced_features(X):
    X_enhanced = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    if 'Cultivation Time' in numeric_cols and 'Degradation time' in numeric_cols:
        X_enhanced['Time_Interaction'] = X_enhanced['Cultivation Time'] * X_enhanced['Degradation time']
        X_enhanced['Time_Ratio'] = X_enhanced['Cultivation Time'] / (X_enhanced['Degradation time'] + 1e-8)
    if 'Aperture Size' in numeric_cols and 'Modulus' in numeric_cols:
        X_enhanced['Size_Modulus'] = X_enhanced['Aperture Size'] * X_enhanced['Modulus']
    if 'Aperture Size' in numeric_cols and 'Cultivation Time' in numeric_cols:
        X_enhanced['Size_Time'] = X_enhanced['Aperture Size'] * X_enhanced['Cultivation Time']
    if 'Aperture Size' in numeric_cols:
        X_enhanced['Aperture_Size_Squared'] = X_enhanced['Aperture Size'] ** 2
        X_enhanced['Aperture_Size_Log'] = np.log1p(X_enhanced['Aperture Size'])
    if 'Cultivation Time' in numeric_cols:
        X_enhanced['Cultivation_Time_Sqrt'] = np.sqrt(X_enhanced['Cultivation Time'] + 1e-8)

    categorical_cols = X.select_dtypes(include=['object']).columns
    if 'Material Type' in categorical_cols and 'Functionalization' in categorical_cols:
        X_enhanced['Material_Function'] = X_enhanced['Material Type'].astype(str) + '_' + X_enhanced['Functionalization'].astype(str)
    if 'Polarization' in categorical_cols and 'Crosslink_Method' in categorical_cols:
        X_enhanced['Polar_Crosslink'] = X_enhanced['Polarization'].astype(str) + '_' + X_enhanced['Crosslink_Method'].astype(str)
    
    return X_enhanced

def drop_interactions_if_needed(X):
    if not DROP_INTERACTIONS:
        return X
    X2 = X.copy()
    for pref in INTERACTION_PREFIXES:
        cols = [c for c in X2.columns if c == pref or c.startswith(pref + '_')]
        if cols:
            X2.drop(columns=cols, inplace=True, errors='ignore')
    return X2

def safe_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))

def build_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

# ========================
# Main Routine
# ========================
def main():
    print("IL-10 Multi-Seed MLP + TTR Model Evaluation...\n")

    df = pd.read_excel(r"C:\\Users\\Cherry\\Desktop\\PNG\\IL10-3.xlsx")
    df.columns = df.columns.str.strip()
    if 'IL-10' not in df.columns:
        raise ValueError("Excel file must contain column 'IL-10'")

    X_raw = df.drop(columns=['IL-10'])
    y_raw = pd.to_numeric(df['IL-10'], errors='coerce').astype(float)

    results = []

    for seed in SEEDS:
        print(f"▶ Running with seed: {seed}")
        X = create_advanced_features(X_raw)
        X = drop_interactions_if_needed(X)
        y = y_raw.copy()

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

        ohe = build_ohe()
        cat_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', ohe)
        ])
        num_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('scaler', StandardScaler())
        ])
        pre = ColumnTransformer([
            ('num', num_tf, num_cols),
            ('cat', cat_tf, cat_cols)
        ])

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

        if APPLY_WINSORIZE_Y:
            hi = np.quantile(y_tr, WINSOR_HIGH_Q)
            y_tr = np.clip(y_tr, a_min=None, a_max=hi)

        if Y_TRANSFORM == 'quantile':
            y_transformer = QuantileTransformer(output_distribution='normal', random_state=seed)
            y_eval_tf = QuantileTransformer(output_distribution='normal', random_state=seed)
        else:
            y_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            y_eval_tf = PowerTransformer(method='yeo-johnson', standardize=True)

        base_mlp = MLPRegressor(
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            max_iter=1500,
            verbose=False
        )

        model = Pipeline([
            ('pre', pre),
            ('reg', TransformedTargetRegressor(
                regressor=base_mlp,
                transformer=y_transformer
            ))
        ])

        param_grid = {
            'reg__regressor__hidden_layer_sizes': [
                (128, 64), (256, 128), (256, 128, 64), (384, 192, 96)
            ],
            'reg__regressor__activation': ['relu'],
            'reg__regressor__alpha': [1e-4, 5e-4, 1e-3, 3e-3],
            'reg__regressor__learning_rate_init': [3e-4, 5e-4, 1e-3],
            'reg__regressor__batch_size': [32, 64, 128],
            'reg__regressor__solver': ['adam'],
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        gscv = GridSearchCV(model, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0)
        gscv.fit(X_tr, y_tr)
        best_model = gscv.best_estimator_

        yhat_tr = best_model.predict(X_tr)
        yhat_te = best_model.predict(X_te)

        r2_tr = r2_score(y_tr, yhat_tr)
        r2_te = r2_score(y_te, yhat_te)
        mse_te = mean_squared_error(y_te, yhat_te)
        mae_te = mean_absolute_error(y_te, yhat_te)
        mape_te = safe_mape(y_te, yhat_te)

        # Evaluate in transformed space (“log/transformed scale”)
        y_eval_tf.fit(y_tr.values.reshape(-1, 1))
        r2_te_tf = r2_score(
            y_eval_tf.transform(y_te.values.reshape(-1, 1)).ravel(),
            y_eval_tf.transform(yhat_te.reshape(-1, 1)).ravel()
        )

        # === Per-seed Results ===
        print(f"R² on test set (log/transformed scale): {r2_te_tf:.4f}")
        print(f"R² on training set (original scale): {r2_tr:.4f}")
        print(f"R² on test set (original scale): {r2_te:.4f}")
        print(f"Test MSE: {mse_te:.2f}")
        print(f"Test MAE: {mae_te:.2f}")
        print(f"Test MAPE (%): {mape_te * 100:.2f}")
        print(f"Best Parameters: {gscv.best_params_}")

        results.append({
            'seed': seed,
            'r2_train': r2_tr,
            'r2_test': r2_te,
            'r2_test_log': r2_te_tf,
            'mse_test': mse_te,
            'mae_test': mae_te,
            'mape_test_%': mape_te * 100,
            'best_params': str(gscv.best_params_)
        })

    # === Final Aggregated Results ===
    df_results = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print("Final IL-10 Multi-Seed MLP + TTR Model Results")
    print(f"{'='*50}")
    print(f"Average Test R² (original scale): {df_results['r2_test'].mean():.4f} ± {df_results['r2_test'].std():.4f}")
    print(f"Average Test R² (log scale): {df_results['r2_test_log'].mean():.4f} ± {df_results['r2_test_log'].std():.4f}")
    print(f"Average Test MSE: {df_results['mse_test'].mean():.2f} ± {df_results['mse_test'].std():.2f}")
    print(f"Average Test MAE: {df_results['mae_test'].mean():.2f} ± {df_results['mae_test'].std():.2f}")
    print(f"Average Test MAPE (%): {df_results['mape_test_%'].mean():.2f} ± {df_results['mape_test_%'].std():.2f}")

    # Save results
    df_results.to_csv("il10_final_results_mlp.csv", index=False)
    print("\nResults saved to il10_final_results_mlp.csv")


if __name__ == "__main__":
    main()
