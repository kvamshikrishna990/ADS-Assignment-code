# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:54:04 2023

@author: k vamshi krishna
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Reading the CSV file into ap
ap = pd.read_csv('Ads_2.csv')
print(ap.head(11))


print(ap.describe())
print(ap.transpose())


# changing  old column names to new column names
column_name = {
    '1990 [YR1990]': '1990',
    '2000 [YR2000]': '2000',
    '2013 [YR2013]': '2013',
    '2014 [YR2014]': '2014',
    '2015 [YR2015]': '2015',
    '2016 [YR2016]': '2016',
    '2017 [YR2017]': '2017',
    '2018 [YR2018]': '2018',
    '2019 [YR2019]': '2019',
    '2020 [YR2020]': '2020',
    '2021 [YR2021]': '2021',
    '2022 [YR2022]': '2022'
}

# Renaming the columns 
ap.rename(columns=column_name, inplace=True)


ap.rename(columns=column_name, inplace=True)

# Filter the DataFrame based on the specified criteria
series_code1 = 'SP.POP.TOTL'
countries = ['AFG', 'BGD', 'IND',  'NPL', 'PAK', ]
start_year = 2017

# Select columns for the desired range of 5 years
desired_years = list(range(start_year, start_year + 5))
columns_for_years = [f'{year}' for year in desired_years]

filtered_df1 = ap.loc[(ap['Series Code'] == series_code1) &
                      (ap['Country Code'].isin(countries)),
                      ['Country Name', 'Series Code'] + columns_for_years]

filtered_df1 = filtered_df1.iloc[:, 0:]


# Replace with the actual 'Series Code'
series_code2 = 'AG.LND.IRIG.AG.ZS'
countries = ['AFG', 'BGD', 'IND', 'NPL', 'PAK',]
start_year = 2017

# Select columns for the  range of 5 years
desired_years = list(range(start_year, start_year + 5))
columns_for_years = [f'{year}' for year in desired_years]

filtered_df2 = ap.loc[(ap['Series Code'] == series_code2) &
                      (ap['Country Code'].isin(countries)),
                      ['Country Name', 'Series Code'] + columns_for_years]
filtered_df2 = filtered_df2.iloc[:, 0:]

# Taking only  five countrys for plots
countries = ['Afghanistan', 'Bangladesh', 'India', 'Nepal', 'Pakistan']
# Taking only five years
years = [2017, 2018, 2019, 2020, 2021]


# Defining population data
population_data = {
    'Afghanistan': [35643418, 36296400, 37769499, 38972230, 40099462],
    'Bangladesh': [161793964, 165303498, 165516222, 167420951, 169356251],
    'India': [1354195680, 1372790000, 1383112050, 1396387127, 1407563842],
    'Nepal': [28183426, 28608715, 28832496, 29348627, 30034989],
    'Pakistan': [216379655, 220892331, 223293280, 227196741, 231402117]
}


# Define agricultural irrigation data
irrigation_data = {
    'Afghanistan': [5.990503825, 5.122336227, 6.006314128, 6.506929763, 6.506929763],
    'Bangladesh': [57.80576552, 57.71726237, 78.77836305, 79.3614021, 78.9034565],
    'India': [38.30101211, 38.789308, 39.96421599, 42.2658231, 42.26577835],
    # Replace with actual values
    'Nepal': [np.nan, np.nan, np.nan, np.nan, np.nan],
    # Replace with actual values
    'Pakistan': [49.23925087, 50.41539253, 50.4638934, 52.66454266, np.nan]
}


def plot_irrigation_data(irrigation_data, years, colormap='viridis'):
    """
    Bar Plots agricultural irrigation data for multiple countries over the specified years.

    Parameters:
     irrigation_data: containing irrigation data for different countries.
     years : List of years to include in the plot.
     colormap : Colormap is to use colours  for bar graph.


    """
    # Create a DataFrame from the irrigation_data
    df_irrigation = pd.DataFrame(irrigation_data, index=years)

    # Ploting the bar graph
    plt.figure(figsize=(10, 6))
    df_irrigation.plot(kind='bar', width=0.8, colormap=colormap)

    # Adding labels and titles
    plt.xlabel('Year')
    plt.ylabel('Percentage of Agricultural Irrigation')
    plt.title('Agricultural Irrigation Data ({}-{})'.format(min(years), max(years)))

    # Show the plot
    plt.legend(title='Country')
    plt.show()





def plot_irrigation_line_chart(irrigation_data, countries, years):
    """
    Plots a line chart for agricultural irrigation area percentage over the specified years.

    Parameters:
     irrigation_data :  containing irrigation data for different countries.
     countries : List of countries to include in the plot.
     years : List of years to include in the plot.

    """
    # Create a size of figure
    plt.figure(figsize = (10, 6))

    # Plot a line for each country
    for country in countries :
        plt.plot(years , irrigation_data[country], label = country , marker ='o')

    # Adding labels and title
    plt.title(
        'Agricultural Irrigation Area (%) Over Years ({}-{})'.format(min(years), max(years)))
    plt.xlabel('Year')
    plt.ylabel('Agricultural Irrigation Area (%)')

    # Adding legend
    plt.legend()

    # used for Showing the grid
    plt.grid(True)

    # Display the plot
    plt.show()





def plot_population_pie_chart(data, selected_year):
    """
    Plots a pie chart for the population distribution for the selected year of 2021.

    Parameters:
     data :  containing population data for different countries.
     selected_year : The selected year for which to plot the pie chart.

    """

    # Selecting data from the selected year
    data_for_selected_year = data[[
        'Country Name', 'Series Code', selected_year]]

    # Set a required color palette
    colors = sns.color_palette("pastel")

    # Draw a pie chart for the selected year of 2021.
    plt.figure(figsize=(10, 8))
    plt.pie(data_for_selected_year[selected_year], labels=data_for_selected_year['Country Name'],
            autopct = '%1.1f%%', startangle = 90, colors = colors, wedgeprops = dict(width = 0.4))

    # Adding a circle in the  center to make like a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title(f'Population Distribution for {selected_year}', fontsize=16)
    plt.show()





def plot_population_heatmap(countries, years, population_data, color_palette="coolwarm"):
    """
    Plots a population heatmap for multiple countries over years.

    Parameters:
     countries : List of country names.
     years : List of years.
     population_data :  containing population data for each country and year.
     color_palette : Seaborn color palette for the heatmap.

    """
    # Combine population and agricultural land data into a single DataFrame
    combined_df = pd.DataFrame({
        'Country': countries * len(years),
        'Year': sorted(years * len(countries)),
        'Population': [population_data[country][i] for country in countries for i in range(len(years))]
    })

    # Pivot the DataFrame for the heatmap
    heatmap_data = combined_df.pivot(
        index='Country', columns='Year', values='Population')

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a custom color palette
    colors = sns.color_palette(color_palette, as_cmap=True)

    # Creating a heatmap using Seaborn with differnt colours
    plt.figure(figsize = (12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap=colors, fmt='g',
                linewidths=.5, cbar_kws={'label': 'Population'})

    # Adding labels and title
    plt.xlabel('Years')
    plt.ylabel('Countrys')
    plt.title('Population Heatmap ({}-{})'.format(min(years), max(years)))

    plt.show()





def create_dataframe_for_country(country, years, data, column_name):
    df = pd.DataFrame({
        'Country': [country] * len(years),
        'Year': sorted(years),
        column_name: [data[country][i] for i in range(len(years))]
    })
    return df

# Function to plot box plots for a specific country


def plot_boxplots_for_country(country_df, column, ylabel, title):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Country', y=column, data=country_df)
    plt.xlabel('Country')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Using the function to plot bar graph
plot_irrigation_data(irrigation_data, years)

# Using the function to plot line chart
plot_irrigation_line_chart(irrigation_data, countries, years)

# Using the function to plot pie chart
plot_population_pie_chart(filtered_df1, '2021')

# Using the function to plot heat map
plot_population_heatmap(
    countries, years, population_data, color_palette="coolwarm")


# Given data
population_data = {
    'India': [1354195680, 1372790000, 1383112050, 1396387127, 1407563842]
}

# Create DataFrame for population in India
population_df_india = create_dataframe_for_country(
    'India', years, population_data, 'Population')

# Plot box plot for population in India
plot_boxplots_for_country(population_df_india, 'Population',
                          'Population', 'Population Box Plots for India (2017-2021)')
