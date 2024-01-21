# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 00:33:23 2024

@author: k vamshi krishna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import curve_fit
from scipy.stats import t


def load_data(file_path):
    """
    Load data from a CSV file and drop rows with missing values.

    """
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    return data.dropna()


def apply_kmeans(data, columns1, n_clusters=3):
    """
    Apply KMeans clustering to the specified columns of the data.

    Parameters:
    - data (pd.DataFrame): Input data.
    - columns1 (list): List of columns to use for clustering.
    - n_clusters (int): Number of clusters for KMeans.

    Returns:
    - Tuple[pd.DataFrame, KMeans]: Data with cluster labels and KMeans model.
    """
    clustering = data[columns1]
    scaler = StandardScaler()
    norm = scaler.fit_transform(clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(norm)
    data["Cluster"] = kmeans.labels_
    return data, kmeans


def visualize_clusters(data, x, y, filter_year=2000):
    """
    Visualize clusters based on specified x and y columns.

    Parameters:
    - data: Input data with cluster labels.
    - x : X-axis column for visualization.
    - y : Y-axis column for visualization.
    - filter_year : Filter data based on the created year.

    """
    filtered_data = data[data['created_year'] >= filter_year]

    plt.figure(figsize=(10, 8))
    for cluster, group in filtered_data.groupby("Cluster"):
        plt.scatter(group[x], group[y], label="Cluster {}".format(cluster))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} and {y} from {filter_year}")
    plt.legend()
    plt.show()


def excel(data, excel_file_path='Statistics.xlsx'):
    """
    Write selected columns of data to an Excel file.

    Parameters:
    - excel_file_path : Path to the Excel file.
    """
    selected_columns = ['created_year', 'subscribers',
                        'Youtuber', 'uploads', 'video views']
    try:
        data[selected_columns].to_excel(excel_file_path, index=False)
        print(f'Selected columns written to {excel_file_path}')
    except Exception as e:
        print(f'Error writing to Excel file: {e}')


def fit_and_visualize1(data, country_filter, x_column, y_column, future_youtubers=[]):
    """
    Fit a linear curve to data and visualize with confidence interval.

    Parameters:
    - country_filter: Country to filter data.
    - x_column: X-axis column for fitting.
    - y_column : Y-axis column for fitting.
    - future_youtubers: List of future youtubers for prediction.
    """
    filtered_data = data[data['Country'] == country_filter]
    label_encoder = LabelEncoder()
    filtered_data['Youtuber_encoded'] = label_encoder.fit_transform(
        filtered_data['Youtuber'])

    x = filtered_data['Youtuber_encoded'].values
    y = filtered_data[y_column].values

    popt, pcov = curve_fit(linear_func, x, y, p0=[1, 0], maxfev=100000000)

    x_line, y_fit, err_lower, err_upper = err_ranges(x, popt, pcov)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, label="Data", color='blue')
    plt.plot(x_line, y_fit, label="Linear Fit", color='red')
    plt.fill_between(x_line, err_lower, err_upper,
                     color='orange', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Youtuber (Encoded)')
    plt.ylabel(y_column)
    plt.title(f" {country_filter} - {y_column} based on Youtuber")

    if future_youtubers:
        encoded_future_youtubers = label_encoder.transform(
            [fy for fy in future_youtubers if fy in label_encoder.classes_])
        predicted_future_years = linear_func(encoded_future_youtubers, *popt)
        plt.scatter(encoded_future_youtubers, predicted_future_years,
                    color='green', marker='o', label='Future Prediction')

    plt.legend()
    plt.show()


def fit_and_visualize2(data, country_filter, x_column, y_column, future_youtubers=[]):
    """
    Fit a linear curve to data and visualize with confidence interval.

    Parameters:
    - country_filter : Country to filter data.
    - x_column : X-axis column for fitting.
    - y_column: Y-axis column for fitting.
    - future_youtubers : List of future youtubers for prediction.
    """
    filtered_data = data[data['Country'] == country_filter]
    label_encoder = LabelEncoder()
    filtered_data['Youtuber_encoded'] = label_encoder.fit_transform(
        filtered_data['Youtuber'])

    x = filtered_data[x_column].values
    y = filtered_data[y_column].values

    popt, pcov = curve_fit(linear_func, x, y, p0=[1, 0], maxfev=100000000)

    x_line, y_fit, err_lower, err_upper = err_ranges(x, popt, pcov)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, label="Data", color='red')
    plt.plot(x_line, y_fit, label="Linear Fit", color='black')
    plt.fill_between(x_line, err_lower, err_upper,
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f" {country_filter} - {y_column} based on {x_column}")

    if future_youtubers:
        encoded_future_youtubers = label_encoder.transform(
            [fy for fy in future_youtubers if fy in label_encoder.classes_])
        predicted_future_values = linear_func(encoded_future_youtubers, *popt)
        plt.scatter(encoded_future_youtubers, predicted_future_values,
                    color='green', marker='o', label='Future Prediction')

    plt.legend()
    plt.show()


def clean_subscribers(subscribers):
    try:
        return float(subscribers)
    except ValueError:
        return np.nan


def fit_and_visualize3(data, country_filter, x_column, y_column, future_youtubers=[]):
    """
    Fit a linear curve to data and visualize with confidence interval.

    Parameters:
    - country_filter : Country to filter data.
    - x_column : X-axis column for fitting.
    - y_column: Y-axis column for fitting.
    - future_youtubers : List of future youtubers for prediction.
    """
    filtered_data = data[data['Country'] == country_filter]
    label_encoder = LabelEncoder()
    filtered_data['Youtuber_encoded'] = label_encoder.fit_transform(
        filtered_data['Youtuber'])

    # Clean 'subscribers' column
    filtered_data['subscribers'] = filtered_data['subscribers'].apply(
        clean_subscribers)

    # Remove rows with NaN in 'subscribers' column
    filtered_data = filtered_data.dropna(subset=['subscribers'])

    x = filtered_data[x_column].values
    y = filtered_data[y_column].values

    popt, pcov = curve_fit(linear_func, x, y, p0=[1, 0], maxfev=100000000)

    x_line, y_fit, err_lower, err_upper = err_ranges(x, popt, pcov)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, label="Data", color='orange')
    plt.plot(x_line, y_fit, label="Linear Fit", color='gray')
    plt.fill_between(x_line, err_lower, err_upper,
                     color='green', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f" {country_filter} - {y_column} based on {x_column}")

    if future_youtubers:
        encoded_future_youtubers = label_encoder.transform(
            [fy for fy in future_youtubers if fy in label_encoder.classes_])
        predicted_future_values = linear_func(encoded_future_youtubers, *popt)
        plt.scatter(encoded_future_youtubers, predicted_future_values,
                    color='green', marker='o', label='Future Prediction')

    plt.legend()
    plt.show()


def linear_func(x, a, b):
    """
    Linear function for curve fitting.

    Parameters:
    - x : Independent variable.
    - a : Slope parameter.
    - b : Intercept parameter.
    """
    return a * x + b


def err_ranges(x, popt, pcov, alpha=0.05):
    """
    Calculate error ranges for the fitted curve.

    Parameters:
    - x : Independent variable.
    - popt: Optimal values for the parameters so that the sum of the squared residuals is minimized.
    - pcov: The estimated covariance of popt.
    - alpha: Significance level for confidence interval.
    """
    tval = t.ppf(1.0 - alpha / 2., len(x) - len(popt))

    x_line = np.linspace(min(x), max(x), 100)
    y_fit = linear_func(x_line, *popt)
    SE = np.sqrt(np.diag(pcov))

    # The standard error of prediction at each x value
    y_err = np.sqrt(SE[0]**2 * x_line + SE[1]**2)

    err_lower = y_fit - tval * y_err
    err_upper = y_fit + tval * y_err

    return x_line, y_fit, err_lower, err_upper


# Read the CSV file directly
data = load_data("Global YouTube Statistics.csv")

columns1 = ['subscribers', 'created_year']
data, kmeans_model = apply_kmeans(data, columns1)
visualize_clusters(data, 'created_year', 'subscribers')

fit_and_visualize1(data, 'India', 'Youtuber_encoded',
                   'created_year', future_youtubers=['WWE', 'BLACKPINK'])
fit_and_visualize2(data, 'India', 'video views', 'uploads',
                   future_youtubers=['WWE', 'BLACKPINK'])
fit_and_visualize3(data, 'India', 'video views', 'subscribers',
                   future_youtubers=['WWE', 'BLACKPINK'])

excel(data, excel_file_path='Statistics.xlsx')
