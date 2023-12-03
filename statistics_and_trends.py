import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv_file(name):
    '''
    Read data from a CSV file, perform preprocessing, and return years as columns and countries as an index.
    
    Parameters:
        name (str): The name of the CSV file.
        
    Returns:
        pd.DataFrame: Transposed DataFrame with years as columns and countries as an index.
        pd.DataFrame: Original data DataFrame.
    '''
    # Read csv file and set variables
    df = pd.read_csv(name, skiprows=4)
    df.drop(columns=['Country Code', 'Indicator Code'], axis=1, inplace=True)
    years = df.head(0).drop(
        ['Country Name', 'Indicator Name', 'Unnamed: 67'], axis=1)
    return years.T, df


def get_indicator_data(indicator, data):
    '''
    Extract the data for a specific indicator from the World Bank dataset.
    
    Parameters:
        indicator (str): The name of the indicator.
        data (pd.DataFrame): The original data DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame containing data for the specified indicator.
    '''
    return data[data['Indicator Name'] == indicator]


def line_graph(countries):
    '''
    Create a line graph for specified countries over the years for a given indicator.
    
    Parameters:
        countries (pd.DataFrame): Indicator data for specified countries.
    '''
    cols = ['Country Name', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
            '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    for indicator_name in ['Population, total', 'Access to electricity (% of population)']:
        data = get_indicator_data(indicator_name, countries)
        data = data[data['Country Name'].isin(
            ['Australia', 'Pakistan', 'India', 'China', 'Bangladesh', 'United States'])]
        data = data[cols]
        data.set_index('Country Name', inplace=True)

        plt.figure(figsize=(10, 6))
        for country in data.index:
            plt.plot(data.columns, data.loc[country],
                     label=country, marker='o')

        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title(indicator_name.upper())
        plt.xticks(rotation=45)
        plt.xticks(data.columns, rotation=45,
                   ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.legend()
        plt.show()


def bar_plot(countries):
    '''
    Create a bar plot for specified countries for a given indicator.
    
    Parameters:
        countries (pd.DataFrame): Indicator data for specified countries.
    '''
    cols = ['Country Name', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
            '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    for indicator_name in ['CO2 emissions (kt)', 'Forest area (sq. km)']:
        data = get_indicator_data(indicator_name, countries)
        data = data[data['Country Name'].isin(
            ['Pakistan', 'India', 'China', 'Australia', 'Bangladesh', 'United States'])]
        data = data[cols]
        data.set_index('Country Name', inplace=True)

        plt.figure(figsize=(10, 6))

        # Set width of each bar
        bar_width = 0.15

        for i, country in enumerate(data.index):
            # Calculate the position for each bar
            position = [
                pos + i * bar_width for pos in range(len(data.columns))]
            plt.bar(position, data.loc[country],
                    width=bar_width, label=country)

        # Calculate the position for x-ticks (centered between the bars)
        tick_positions = [pos + 0.5 * bar_width *
                          (len(data.index) - 1) for pos in range(len(data.columns))]

        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title(indicator_name.upper())
        plt.xticks(tick_positions, data.columns, rotation=45,
                   ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.legend()
        plt.show()


def heatmap(data):
    '''
    Display a heatmap graph showing the correlation between different indicators causing global warming.
    
    Parameters:
        data (pd.DataFrame): Original data DataFrame.
    '''
    for country_name in ['Pakistan', 'India', 'United States']:
        # Get country-specific data
        country_data = data[data['Country Name'] == country_name]

        # Define an empty DataFrame
        indicator_data = pd.DataFrame()

        # List of indicators to include in the heatmap
        indicators_to_include = [
            "Access to electricity (% of population)",
            "CO2 emissions (kt)",
            "Electric power consumption (kWh per capita)",
            "Population, total",
            "Forest area (sq. km)"
        ]

        # Populate the DataFrame with indicator data
        for indicator_name in indicators_to_include:
            indicator_data[indicator_name] = country_data[country_data['Indicator Name']
                                                          == indicator_name].iloc[:, 2:].values.flatten()

        # Drop unnecessary columns and reset index
        indicator_data = indicator_data.drop(
            ['Country Name', 'Indicator Name'], errors='ignore').reset_index(drop=True)

        # Plotting the data
        ax = plt.axes()

        # Use seaborn to plot heatmap
        sns.heatmap(indicator_data.corr(), cmap="YlGnBu", annot=True, ax=ax)

        # Adding title
        ax.set_title(country_name)

        # Show the plot
        plt.show()


# Main Function
if __name__ == "__main__":
    # Set Variables
    _, countries = read_csv_file('API_19_DS2_en_csv_v2_4756035.csv')

    # # Draw line graph
    line_graph(countries)

    # Draw bar plot
    bar_plot(countries)

    # Draw heatmap
    heatmap(countries)
