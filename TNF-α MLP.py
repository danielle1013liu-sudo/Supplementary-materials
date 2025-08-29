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

# Configuration
SEEDS = [42, 793, 1579]
Y_TRANSFORM = 'quantile'
APPLY_WINSORIZE_Y = True
WINSOR_HIGH_Q = 0.995
DROP_INTERACTIONS = True

INTERACTION_PREFIXES = [
    'Time_Interaction', 'Time_Ratio',
    'Cultivation_Aperture', 'Degradation_Modulus',
    'Material_Macrophage', 'Function_Crosslink',
]

# Utility functions
def create_advanced_features(X):
    X = X.copy()
    num = X.select_dtypes(include=[np.number]).columns
    if 'Cultivation Time' in num and 'Degradation time' in num:
        X['Time_Interaction'] = X['Cultivation Time'] * X['Degradation time']
        X['Time_Ratio'] = X['Cultivation Time'] / (X['Degradation time'] + 1e-8)
    if 'Cultivation Time' in num and 'Aperture Size' in num:
        X['Cultivation_Aperture'] = X['Cultivation Time'] * X['Aperture Size']
    if 'Degradation time' in num and 'Modulus' in num:
        X['Degradation_Modulus'] = X['Degradation time'] * X['Modulus']
    if 'Cultivation Time' in num:
        X['Cultivation_Time_Squared'] = X['Cultivation Time'] ** 2
        X['Cultivation_Time_Log'] = np.log1p(X['Cultivation Time'])
    if 'Degradation time' in num:
        X['Degradation_Time_Sqrt'] = np.sqrt(X['Degradation time'] + 1e-8)
    if 'Aperture Size' in num:
        X['Aperture_Size_Log'] = np.log1p(X['Aperture Size'])
    cat = X.select_dtypes(include=['object']).columns
    if 'Material Type' in cat and 'Macrophage_Type' in cat:
        X['Material_Macrophage'] = X['Material Type'].astype(str) + '_' + X['Macrophage_Type'].astype(str)
    if 'Functionalization' in cat and 'Crosslink_Method' in cat:
        X['Function_Crosslink'] = X['Functionalization'].astype(str) + '_' + X['Crosslink_Method'].astype(str)
    return X

def drop_interactions_if_needed(X):
    if not DROP_INTERACTIONS:
        return X
    X2 = X.copy()
    for pref in INTERACTION_PREFIXES:
        cols = [c for c in X2.columns if c == pref or c.startswith(pref + '_')]
        X2.drop(columns=cols, inplace=True, errors='ignore')
    return X2

def safe_mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))

def main():
    df = pd.read_excel(r"C:\\Users\\Cherry\\Desktop\\PNG\\TNFA3.xlsx")
    df.columns = df.columns.str.strip()
    X_raw = df.drop(columns=['TNF-α'])
    y_raw = df['TNF-α'].astype(float)

    results = []

    for seed in SEEDS:
        print(f"\nRunning with random seed: {seed}")
        X = create_advanced_features(X_raw)
        X = drop_interactions_if_needed(X)
        y = y_raw.copy()

        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

        cat_tf = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
            'reg__regressor__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
            'reg__regressor__activation': ['relu'],
            'reg__regressor__alpha': [1e-4, 5e-4, 1e-3],
            'reg__regressor__learning_rate_init': [3e-4, 5e-4],
            'reg__regressor__batch_size': [32, 64],
            'reg__regressor__solver': ['adam']
        }

        gscv = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=seed),
                            scoring='r2', n_jobs=-1)
        gscv.fit(X_tr, y_tr)
        best_model = gscv.best_estimator_

        yhat_tr = best_model.predict(X_tr)
        yhat_te = best_model.predict(X_te)

        r2_tr = r2_score(y_tr, yhat_tr)
        r2_te = r2_score(y_te, yhat_te)
        mse_te = mean_squared_error(y_te, yhat_te)
        mae_te = mean_absolute_error(y_te, yhat_te)
        mape_te = safe_mape(y_te, yhat_te)

        y_eval_tf.fit(y_tr.values.reshape(-1,1))
        r2_te_tf = r2_score(
            y_eval_tf.transform(y_te.values.reshape(-1,1)).ravel(),
            y_eval_tf.transform(yhat_te.reshape(-1,1)).ravel()
        )

        print(f"R² on test set (log scale): {r2_te_tf:.4f}")
        print(f"R² on training set (original scale): {r2_tr:.4f}")
        print(f"R² on test set (original scale): {r2_te:.4f}")
        print(f"Test MSE: {mse_te:.2f}")
        print(f"Test MAE: {mae_te:.2f}")
        print(f"Test MAPE (%): {mape_te*100:.2f}")

        results.append({
            'Seed': seed,
            'R2_test': r2_te,
            'MSE': mse_te,
            'MAE': mae_te,
            'MAPE': mape_te * 100
        })

    df_res = pd.DataFrame(results)
    print("\n========= Mean ± Std (Test Set, Original Scale) =========")
    print(df_res.describe().loc[['mean', 'std']].round(2).T)

if __name__ == "__main__":
    main()
