import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression, RandomizedLogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline # Streaming pipelines
import warnings
warnings.filterwarnings("ignore")

pipelines = []

seed = 7

pipelines.append(
    ("Scaled_Ridge",
     Pipeline([
         ("Ridge", Ridge(random_state=seed))
     ]))
)

pipelines.append(
    ("Scaled_Lasso",
     Pipeline([
         ("Lasso", Lasso(random_state=seed))
     ]))
)

pipelines.append(
    ("Scaled_SVR",
     Pipeline([
         ("SVR", SVR())
     ])
     )
)

pipelines.append(
    ("Scaled_RF",
     Pipeline([
         ("RF", RandomForestRegressor(random_state=seed))
     ])
     )
)

pipelines.append(
    ("Scaled_ET",
     Pipeline([
         ("ET", ExtraTreesRegressor(random_state=seed))
     ])
     )
)
pipelines.append(
    ("Scaled_BR",
     Pipeline([
         ("BR", BaggingRegressor(random_state=seed))
     ])
     )
)

pipelines.append(
    ("LinearRegression",
     Pipeline([
         ("LinearRegression", LinearRegression())
     ]))
)


n_folds = 10
X_train = pd.read_csv('./data/clean_data.csv')
Y_train = pd.read_csv('./data/result.csv')

scoring = 'r2'
for name, model in pipelines:
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                 scoring=scoring, n_jobs=-1)
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
