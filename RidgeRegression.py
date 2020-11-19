
# Importing the libraries required
import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np


def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Perform cross-validation:
    cv_score = sklearn.model_selection.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,
                                                       scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])

    # Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)



def ridge_regression(train, test, target, IDcol):
    from sklearn.linear_model import LinearRegression, Ridge
    predictors = [x for x in train.columns if x not in [target] + IDcol]
    alg2 = Ridge(alpha=0.05, normalize=True)
    modelfit(alg2, train, test, predictors, target, IDcol, 'ridge.csv')
    coef2 = pd.Series(alg2.coef_, predictors).sort_values()
    coef2.plot(kind='bar', title='Model Coefficients')
    print("Model has been successfully created and trained. The predicted result is in ridge.csv")
