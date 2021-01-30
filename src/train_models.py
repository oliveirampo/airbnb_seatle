import sys
import numpy as np
import pandas as pd
import itertools
import time


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


import utils


def print_test_score(model_name, model, X_test, y_test):
    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.\nThe RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))


# Simple Linear Regression
def model_01(X_train, y_train, X_test, y_test):
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
    # print_test_score(model_name, cv, X_test, y_test)
    print('DONE\n')


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
    print_test_score(model_name, clf.best_estimator_, X_test, y_test)
    # print('DONE\n')


# Kernel Ridge Regression with grid search of the best parameters and cross-validation
def model_03(X_train, y_train, X_test, y_test):
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
    print_test_score(model_name, clf.best_estimator_, X_test, y_test)


def model_04(X_train, y_train, X_test, y_test):
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
    print_test_score(model_name, model.best_estimator_, X_test, y_test)


def model_05(X_train, y_train, X_test, y_test):
    t1 = time.time()

    model_name = 'MLPRegressor'

    # scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(random_state=1, max_iter=500)
    model.fit(X_train, y_train)

    y_train_preds = model.predict(X_train)
    print("The r^2 score for {} on the training data was {} on {} values. The RMSE was {}".format(
        model_name, r2_score(y_train, y_train_preds), len(y_train), mean_squared_error(y_train, y_train_preds)))
    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.  The RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))

    t2 = time.time()
    time_taken = (t2 - t1) / 60.0
    print('Time taken: {}'.format(time_taken))


# It is too slow to combine so many features
def get_combination_of_feature(key_words):
    # add price at the end
    main_options = ["neighbourhood_cleansed", "property_type", "accommodates", "bedrooms", "review_scores_rating",
                    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_communication",
                    "review_scores_location", "review_scores_value", "instant_bookable", "reviews_per_month",
                    "host_is_superhost", "room_type", "number_of_reviews"]
    weak = ["room_type", "number_of_reviews", "instant_bookable", "host_is_superhost", "review_scores_value",
            "review_scores_communication", "review_scores_cleanliness", "review_scores_accuracy"]
    main_options = ["bedrooms", "accommodates", "reviews_per_month", "review_scores_rating",
                    "neighbourhood_cleansed", "property_type"]

    for col in key_words:
        main_options.append(col)

    features_combination = []
    N = len(main_options)
    # N = 2

    for num_entries in range(N, N + 1):
        combinations = itertools.combinations(main_options, num_entries)
        for values in list(combinations):
            values = (list(values))
            values.append('price')
            features_combination.append(values)

    # for feature in features_combination:
    #     print(feature)
    return features_combination


def main():
    df_listing_detailed = pd.read_csv('inp/listings_detailed.csv')
    df_listing_detailed['price'] = df_listing_detailed['price'].str.replace("[$, ]", "").astype("float")

    print('Reading amenities.')
    key_words = utils.read_key_words('inp/key_words.txt')
    selected_key_words = utils.add_amenities(df_listing_detailed, key_words)
    # print(selected_key_words)

    print('Getting combination of features.')
    features_combination = get_combination_of_feature(selected_key_words)

    print('Training models.')
    count = len(features_combination)
    for features in features_combination:
        print('\n\nFeatures ({}):\n{}\n\n'.format(count, features))
        count = count - 1
        filtered_df = df_listing_detailed[features]

        X, y = utils.clean_data(filtered_df.copy())
        utils.check_null_columns(X.copy())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=123)

        print('\nModel 01')
        model_01(X_train, y_train, X_test, y_test)
        print('\nModel 02')
        model_02(X_train, y_train, X_test, y_test)
        print('\nModel 03')
        model_03(X_train, y_train, X_test, y_test)
        print('\nModel 04')
        model_04(X_train, y_train, X_test, y_test)
        print('\nModel 05')
        model_05(X_train, y_train, X_test, y_test)
        # # sys.exit(123)


if __name__ == "__main__":
    main()
