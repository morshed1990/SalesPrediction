
# Importing the libraries required
import pandas as pd
# import numpy as np
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




from sklearn.ensemble import RandomForestRegressor


def random_forest(train, test, target, IDcol):
    predictors = [x for x in train.columns if x not in [target] + IDcol]
    alg5 = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=4)
    modelfit(alg5, train, test, predictors, target, IDcol, 'ranForest_1.csv')
    coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
    coef5.plot(kind='bar', title='Feature Importances')

    print("Model has been successfully created and trained. The predicted result is in ranForest_1.csv")

    predictors = [x for x in train.columns if x not in [target] + IDcol]
    alg6 = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=100, n_jobs=4)
    modelfit(alg6, train, test, predictors, target, IDcol, 'ranForest_2.csv')
    coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
    coef6.plot(kind='bar', title='Feature Importances')
    print("Model has been successfully created and trained. The predicted result is in ranForest_2.csv")
