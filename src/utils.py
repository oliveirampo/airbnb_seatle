import re
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def plot_box_data(df, attribute, unique_values, title):
    box_data = []

    for val in unique_values:
        trace = go.Box(
            x=df.loc[df[attribute] == val].price.tolist(),
            name=val)
        box_data.append(trace)

    box_layout = go.Layout(xaxis=dict(title='Listing Price'), title=title)
    box_fig = go.Figure(data=box_data, layout=box_layout)
    return box_fig


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe

    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector

    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:

    1. Select the numeric variables in the dataset
    2. Imputes the mean to fill null values for numeric variables
    3. Select the categorical variables
    4. Creates dummy columns for the categorical variables
    5. Create X as all the columns that are not the Price column
    6. Create y as the Price column

    '''

    # Fill numeric columns with the mean
    num_vars = df.select_dtypes(include=['int','float']).copy().columns
    for col in num_vars:
        df[col].fillna((df[col].mean()), inplace=True)

    # Dummy the categorical variables
    cat_vars = df.select_dtypes(include=['object']).copy().columns
    for var in cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)

    X = df.drop(columns=['price'], axis=1)
    y = df['price']

    return X, y


def read_key_words(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = [re.sub(' +', ' ', row.strip()) for row in lines]
    return lines


def amenities_key_words(df):
    amenities = df['amenities'].unique()

    key_words = []

    for i in range(amenities.shape[0]):
        # remove first and last brackets
        row = amenities[i][1:-1]
        row = row.strip().split(',')

        for i in range(len(row)):
            val = row[i].strip()
            key_words.append(val)

    key_words = sorted(set(key_words))
    for word in key_words:
        print('\t', word)
    print(len(key_words))


def add_amenities(df, key_words):
    # add column for each key word with False as value
    names = []
    for word in key_words:
        col_name = 'amenity_' + word.strip().replace(' ', '_')
        names.append(col_name)
        df[col_name] = False

    for word in key_words:
        col_name = 'amenity_' + word.strip().replace(' ', '_')

        for idx, row in df.iterrows():
            val = row['amenities']

            if word in val:
                df.loc[idx, col_name] = True

    # remove columns which only have False values
    rm_columns = df[names].any()
    # get only columns whose values are always False
    rm_columns = rm_columns[~rm_columns].index

    for col in rm_columns:
        df = df.drop(col, axis=1)
        names.remove(col)

    return names


def check_null_columns(df):
    df_null = pd.DataFrame(np.sum(df.isnull()), columns=['null_count'])
    # df_null = df_null[df_null['null_count'] == df.shape[0]]
    df_null = df_null[df_null['null_count'] > 0]
    if df_null.shape[0] != 0:
        print(df_null)
        print('Nan values found.')
        sys.exit(123)


def print_score(model_name, model, X_train, y_train, X_test, y_test):
    # Predict and score the model
    y_train_preds = model.predict(X_train)
    print("The r^2 score for {} on the training data was {} on {} values. The RMSE was {}".format(
        model_name, r2_score(y_train, y_train_preds), len(y_train), mean_squared_error(y_train, y_train_preds)))
    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.  The RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))


def print_test_score(model_name, model, X_test, y_test):
    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.\nThe RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))


# Simple Linear Regression
def model_01(X_train, y_train):
    model_name = 'Linear Regression'
    model = LinearRegression(normalize=True)

    # scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # fit the model
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123)
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv, n_jobs=3)

    # evaluate the model
    print('%.2e (%.2e)' % (np.mean(scores), np.std(scores)))

    return model


# Lasso
def model_02(X_train, y_train, X_test, y_test):
    model_name = 'Lasso'
    model = Lasso(normalize=True, warm_start=True)
    parameters = {'alpha': [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]}

    # fit the model
    clf = GridSearchCV(model, parameters, scoring='r2', n_jobs=3, cv=5)
    clf.fit(X_train, y_train)

    # evaluate the model
    best_index = clf.best_index_
    mean = clf.cv_results_['mean_test_score'][best_index]
    std = clf.cv_results_['std_test_score'][best_index]
    print('Best results for test set: {:.2f} ({:.2f})'.format(mean, std))
    print(clf.best_estimator_)

    return clf.best_estimator_


# Kernel Ridge Regression with grid search of the best parameters and cross-validation
def model_03(X_train, y_train):
    model_name = 'KernelRidge'
    model = KernelRidge(alpha=1.0)
    parameters = {'alpha':[0.01, 0.1, 1.0, 10.0, 100.0], 'kernel':('linear', 'rbf', 'polynomial'),
                  'degree': (2, 3, 4, 5)}

    # fit the model
    clf = GridSearchCV(model, parameters, scoring='r2', n_jobs=3, cv=5)
    clf.fit(X_train, y_train)

    # evaluate the model
    best_index = clf.best_index_
    mean = clf.cv_results_['mean_test_score'][best_index]
    std = clf.cv_results_['std_test_score'][best_index]
    print('Best results for test set: {:.2f} ({:.2f})'.format(mean, std))
    print(clf.best_estimator_)

    return clf.best_estimator_


def model_04(X_train, y_train):
    t1 = time.time()
    model_name = 'RandomForestRegressor'
    model = RandomForestRegressor()

    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=6)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    # Create the random grid
    rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    model = RandomizedSearchCV(estimator=model, scoring='r2', n_iter=180, cv=3, n_jobs=3,
                                param_distributions=rm_grid, random_state=123)

    # Fit the random search model
    model.fit(X_train, y_train)

    # evaluate the model
    best_index = model.best_index_
    mean = model.cv_results_['mean_test_score'][best_index]
    std = model.cv_results_['std_test_score'][best_index]
    print('Best results for test set: {:.2f} ({:.2f})'.format(mean, std))
    print(model.best_estimator_)
    t2 = time.time()
    time_taken = (t2 - t1) / 60.0
    print('Time taken: {}'.format(time_taken))

    return model.best_estimator_


def model_05(X_train, y_train):
    model_name = 'MLPRegressor'

    # scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    model = MLPRegressor(random_state=1, max_iter=500)
    model.fit(X_train, y_train)

    y_train_preds = model.predict(X_train)
    print("The r^2 score for {} on the training data was {} on {} values. The RMSE was {}".format(
        model_name, r2_score(y_train, y_train_preds), len(y_train), mean_squared_error(y_train, y_train_preds)))

    return model, scaler
