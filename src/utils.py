import re
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

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


def plot_detailed_histogram(x_variable, x_label, y_label, component_list, label, df, fig_name):
    """
    Plot histogram with stacked components  using plotly.

        Parameters:
            x_variable (str): Name of column in DataFrame used as x-axis
            x_label (str): Label of the x-axis
            y_label (str): Label of the y-axis
            component_list (list): List of values to be selected in the DataFrame
            label (str): Name of column in the DataFrame from which component_list is selected
            df (pandas DataFrame): DataFrame with data
            fig_name (str): figure name

        Returns:
            hist_fig (plotly.graph_objs): figure object

    For all components specified in component_list
    saves all values relative to x_variable and save them into hist_data.
    Plot hist_data in the form of a stacked histogram.
    """
    hist_data = []

    for component in component_list:
        trace = go.Histogram(
            x=df.loc[df[label] == component][x_variable].tolist(),
            opacity=0.75, name=component)
        hist_data.append(trace)

    hist_layout = go.Layout(barmode='stack', xaxis=dict(title=x_label), yaxis=dict(title=y_label))
    hist_fig = go.Figure(data=hist_data, layout=hist_layout, layout_xaxis_range=[0, 1000])

    hist_fig.write_image(fig_name)

    return hist_fig


def plot_box_data(df, attribute, unique_values, x_variable, x_label, title, fig_name):
    """
    Plot box using plotly.

        Parameters:
            df (pandas DataFrame): DataFrame with all data
            attribute (str): Name of column in DataFrame from which data will be selected
            unique_values (list): List of values that restrains data selection.
            x_variable (str): Name of column in DataFrame used as x-axis
            x_label (str): Label of a-xis
            title (str): Title of plot
            fig_name (str or PathLike): A path of the file including the extension to where the figure will be saved

        Returns:
            box_fig (plotly.graph_objs): figure object

    Selects values from the DataFrame 'df'
    from the column = 'attribute' and values specified in 'unique_values'
    and plot values from the column = 'x_variable'.
    """
    box_data = []

    for val in unique_values:
        trace = go.Box(
            x=df.loc[df[attribute] == val][x_variable].tolist(),
            name=val)
        box_data.append(trace)

    box_layout = go.Layout(xaxis=dict(title=x_label), title=title)
    box_fig = go.Figure(data=box_data, layout=box_layout)

    box_fig.write_image(fig_name)

    return box_fig


def plot_stacked_data(x_values, component_list, label, df, title, fig_name):
    """
    Plot box using plotly.

        Parameters:
            x_values (list): Values of x-ticks (they should be compatible with values in df)
            component_list (list): List of values that restraints components to be stacked
            label (str): Name of column in the DataFrame from which component_list is selected
            df (pandas DataFrame): DataFrame with all data
            title (str): Title of plot
            fig_name (str or PathLike): A path of the file including the extension to where the figure will be saved

        Returns:
            stacked_bar_fig (plotly.graph_objs): figure object

    Selects values from the DataFrame 'df'
    specified in 'component_list' and from the column = 'label' (stacked components)
    and plot the count of values with 'x_values' as x-ticks.
    """

    stacked_bar_data = []

    for component in component_list:
        count = df[df[label] == component]['id'].tolist()
        stacked_bar_trace = go.Bar(name=component, x=x_values, y=count)
        stacked_bar_data.append(stacked_bar_trace)

    stacked_bar_layout = go.Layout(title=title,
                                   barmode='stack', yaxis=dict(title='Count'))

    stacked_bar_fig = go.Figure(data=stacked_bar_data, layout=stacked_bar_layout)

    stacked_bar_fig.write_image(fig_name)

    return stacked_bar_fig


def plot_heat_map(filtered_df, x_values, y_values, column_x, column_y, dependent_variable, title, fig_name):
    """
    Plot heat map using plotly.

        Parameters:
            filtered_df (pandas DataFrame): DataFrame with all data
            x_values (list): List that defines which values of column = 'column_x' will be used
            y_values (list): List that defines which values of column = 'column_y' will be used
            column_x (str): Column name in DataFrame for x-axis
            column_y (str): Column name in DataFrame for y-axis
            dependent_variable (str): Column name in DataFrame for z-axis
            title (str): Title of plot
            fig_name (str or PathLike): A path of the file including the extension to where the figure will be saved

        Returns:
            heatmap_fig (plotly.graph_objs): figure object

    Selects values from the DataFrame 'df' specified in
    'x_values' and from the column = 'column_x' for x-axis,
    'y_values' and from the column = 'column_y' for y-axis
    and from the column = 'dependent_variable' for z-axis
    and plot heatmap.
    """
    heatmap_df = pd.DataFrame(columns=[column_y, dependent_variable])
    for y in y_values:
        data = []
        for x in x_values:
            df = filtered_df.loc[(filtered_df[column_y] == y) & (filtered_df[column_x] == x)]

            if df.shape[0] == 0:
                data.append(np.nan)
            else:
                data.append(df[dependent_variable].values[0])

        a_series = pd.Series([y, data], index=heatmap_df.columns)
        heatmap_df = heatmap_df.append(a_series, ignore_index=True)

    heatmap_price = heatmap_df.price.tolist()

    heatmap_trace = go.Heatmap(
        z=heatmap_price,
        x=x_values,
        y=y_values
    )

    heatmap_layout = dict(title=title,
                          xaxis=dict(automargin=True),
                          yaxis=dict(automargin=True)
                          )
    heatmap_fig = go.Figure(data=heatmap_trace, layout=heatmap_layout)

    heatmap_fig.write_image(fig_name)

    return heatmap_fig


def plot_train_vs_test_data(y_hat, y_predicted):
    """
    Plot actual values versus predicted values.

        Parameters:
            y_hat (list): Actual values
            y_predicted (list): Predicted values

        Returns:
            None
    """
    plt.scatter(y_hat, y_predicted)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.plot(np.unique(y_hat), np.poly1d(np.polyfit(y_hat, y_predicted, 1))(np.unique(y_hat)))
    fig = plt.gcf()
    fig.savefig('images/predicted_vs_actual_price.png')


def plot_train_and_test_data(y_train, y_test, y_train_predicted, y_test_predicted):
    """
    Plot residual error of training and test sets.

        Parameters:
            y_train (list): Actual values of training set
            y_test (list): Actual values of test set
            y_train_predicted (list): Predicted values of training set
            y_test_predicted (list): Predicted values of test set

        Returns:
            fig (figure): figure object
    """

    plt.scatter(y_train_predicted, y_train_predicted - y_train,
                c='blue', marker='o', label='Training data')
    plt.scatter(y_test_predicted, y_test_predicted - y_test,
                c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='lower right')
    fig = plt.gcf()
    return fig


def clean_data(df):
    """
    Cleans the DataFrame df for further analysis
        Parameters:
            df (pandas DataFrame): DataFrame with all data

        Returns:
            X (pandas DataFrame):  A matrix holding all of the variables you want to consider when predicting the response
            y (pandas DataFrame):  the corresponding response vector

    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:

    1. Select the numeric variables in the dataset
    2. Imputes the mean to fill null values for numeric variables
    3. Select the categorical variables
    4. Creates dummy columns for the categorical variables
    5. Create X as all the columns that are not the Price column
    6. Create y as the Price column
    """

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
    """
    Read plain text file with key words to be extracted from 'amenities' column.

        Parameters:
            file_name (str): Name of file

        Returns:
            lines (list): list with key words
    """

    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = [re.sub(' +', ' ', row.strip()) for row in lines]
    return lines


def amenities_key_words(df):
    """
    Print all key words in column = 'amenities' from DataFrame = df

        Parameters:
            df (pandas DataFrame): DataFrame with all data

        Returns:
            lines (list): list with key words
    """
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
    """
    Save key words extracted from column = 'amenities' in DataFrame = df
    as new columns of booleans

        Parameters:
            df (pandas DataFrame): DataFrame with all data
            key_words (list): List of key words and new columns of booleans

        Returns:
            names (list): List column names from list of key words which hast at least one row = True

    1. Add a column for each key word with False as values.
    2. For each newly created column, check if its key words in present in the columns = 'amenities'.
    and change its value to True in positive case.
    3. Remove newly created columns with only False entries.
    """


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
    """
    Check if any column has nan values.
    If this is the case stop the program with an error message.

        Parameters:
            df (pandas DataFrame): DataFrame with all data

        Returns:
            None
    """
    df_null = pd.DataFrame(np.sum(df.isnull()), columns=['null_count'])
    df_null = df_null[df_null['null_count'] > 0]
    if df_null.shape[0] != 0:
        print(df_null)
        print('Nan values found.')
        sys.exit(123)


def print_score(model_name, model, X_train, y_train, X_test, y_test):
    """
    Print r^2 score of train and test sets

        Parameters:
            model_name (str): Name of model.
            model (sklearn model): Trained model.
            X_train (array-like): Train data.
            y_train (array-like): Target values for train set.
            X_test (array-like): Test data.
            y_test (array-like): Target values for test set.

        Returns:
            None
    """

    # Predict and score the model
    y_train_preds = model.predict(X_train)
    print("The r^2 score for {} on the training data was {} on {} values. The RMSE was {}".format(
        model_name, r2_score(y_train, y_train_preds), len(y_train), mean_squared_error(y_train, y_train_preds)))
    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.  The RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))


def print_test_score(model_name, model, X_test, y_test):
    """
    Print r^2 score of test set.

        Parameters:
            model_name (str): Name of model.
            model (sklearn model): Trained model.
            X_test (array-like): Test data.
            y_test (array-like): Target values for test set.

        Returns:
            None
    """

    y_test_preds = model.predict(X_test)
    print("The r^2 score for {} on the test data was {} on {} values.\nThe RMSE was {}".format(
        model_name, r2_score(y_test, y_test_preds), len(y_test), mean_squared_error(y_test, y_test_preds)))


def model_01(X_train, y_train):
    """
    Fits ordinary least squares Linear Regression.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Target values.

        Returns:
            model (sklearn model)
    """
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
    print('{}: %.2e (%.2e)' % (model_name, np.mean(scores), np.std(scores)))

    return model


def model_02(X_train, y_train):
    """
    Fits Linear Model trained with L1 prior as regularizer (aka the Lasso).

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Target values.

        Returns:
            model (sklearn model)
    """
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
    print('Best results for test set {}: {:.2f} ({:.2f})'.format(model_name, mean, std))
    print(clf.best_estimator_)

    return clf.best_estimator_


def model_03(X_train, y_train):
    """
    Fits Kernel ridge regression.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Target values.

        Returns:
            model (sklearn model)
    """
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
    print('Best results for test set {}: {:.2f} ({:.2f})'.format(model_name, mean, std))
    print(clf.best_estimator_)

    return clf.best_estimator_


def model_04(X_train, y_train):
    """
    Fits random forest regressor.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Target values.

        Returns:
            model (sklearn model)
    """
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
    print('Best results for test set {}: {:.2f} ({:.2f})'.format(model_name, mean, std))
    print(model.best_estimator_)
    t2 = time.time()
    time_taken = (t2 - t1) / 60.0
    print('Time taken: {}'.format(time_taken))

    return model.best_estimator_


def model_05(X_train, y_train):
    """
    Fits Multi-layer Perceptron regressor.

        Parameters:
            X_train (array-like): Training data.
            y_train (array-like): Target values.

        Returns:
            model (sklearn model)
    """
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
