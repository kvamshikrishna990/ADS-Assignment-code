# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:31:33 2023

@author: k vamshi krishna
"""


import pandas as pd
import matplotlib.pyplot as plt


def plot_covid_data(covid, state_column, infected_column, tested_column, deaths_column):
    
    """
    Plot line graphs for infected, tested, and deaths in each state.

    Parameters:
    - dataset: Pandas DataFrame
    - state_column: Name of the column containing state names
    - infected_column: Name of the column containing infected cases
    - tested_column: Name of the column containing tested cases
    - deaths_column: Name of the column containing deaths
    """

    # Plotting infected,Tested,Deaths cases
    plt.figure(figsize=(20, 8))
    plt.plot(covid[state_column], covid[infected_column],
             marker='o', label='Infected', color='blue')

    plt.plot(covid[state_column], covid[tested_column],
             marker='o', label='Tested', color='orange')

    plt.plot(covid[state_column], covid[deaths_column],
             marker='+', label='Deaths', color='red')

    # Adding labels and title
    plt.xlabel('US States')
    plt.ylabel('Number of Cases(MILLIONS')
    plt.title('COVID-19 Data in Each State of US')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.show()




def plot_topstates(dataset, column_name, n=5):
    Coviddeaths = dataset.sort_values(by=column_name, ascending=False)
    states = Coviddeaths.head(n)
    """ Parameters:
        - dataset: Pandas DataFrame
        - column_name: Name of the column for sorting
        - n: Number of top states to plot (default is 5)
        """


# Plot X and Y axis
    X = list(states.iloc[:, 0])
    Y = list(states.iloc[:, 3])

    plt.bar(X, Y, color='r')
    plt.xlabel('States')
    plt.ylabel('Number of Deaths')
    plt.title(f'Top {n} States with Highest Deaths')
    plt.show()




def plot_topinfected_states_pie_chart(dataset, n=10):
    """
    Plot a pie chart for the distribution of COVID-19 cases across different states.

    Parameters:
    - dataset: Pandas DataFrame
    - column_name: Name of the column to represent in the pie chart
    """

    # Sort the dataset based on infected cases
    Covid = dataset.sort_values(by='Infected', ascending=False)

    # Select the top N infected states
    infected_states = Covid.head(n)

    # Plotting the pie chart
    plt.figure(figsize=(6, 7))
    plt.pie(infected_states['Infected'],
            labels=infected_states['State'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of Infected Cases in Top {n} States')
    plt.axis('equal')
    plt.show()


# Reading the CSV file
covid = pd.read_csv("COVID19_state.csv")

# Using the function to plot line graphs for infected, tested, and deaths
plot_covid_data(covid, 'State', 'Tested', 'Infected', 'Deaths')

# Using the function to plot top 5 states with the most deaths
plot_topstates(covid, 'Deaths', n=5)


# Using the function to plot a pie chart for the distribution of infected cases in the top 10 states
plot_topinfected_states_pie_chart(covid, n=10)
