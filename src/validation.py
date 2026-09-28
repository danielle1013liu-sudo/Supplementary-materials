import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .data import feature_names, RESPONSE, cv_folds

def metrics(observed, predicted):
    return {'R2': float(r2_score(observed, predicted)),
            'RMSE_pg_ml': float(np.sqrt(mean_squared_error(observed, predicted))),
            'MAE_pg_ml': float(mean_absolute_error(observed, predicted))}

def validation_predictions(target, pipeline, training):
    predictions = np.empty(len(training))
    for fit, validation in cv_folds(target):
        model = clone(pipeline)
        model.fit(training.iloc[fit][feature_names()], training.iloc[fit][RESPONSE])
        predictions[validation] = model.predict(training.iloc[validation][feature_names()])
    return predictions
