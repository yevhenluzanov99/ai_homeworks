import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import kendalltau
from itertools import combinations
import re
from scipy.stats import shapiro
import numpy as np

df = pd.read_csv("homework3/homework3_data_preprocessed_result.csv")

def check_lilliefors_normality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check normality of the data.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with normality check results.

    In comparing two method for normality test, we can see that lilliefors method is more accurate than visual method.
    Because that graphics when we can see normal distribution is not always accurate. 
    I transform data to fix distribution using different methods:
        Log-Transformation:
            df[column] = np.log1p(df[column])
        Ranking:
            df[column] = df[column].rank()
    And after ranking method I can see that data has not normal distribution. Check column = bov
    
    """
    # Check normality of the data
    normality_results = {}
    for column in df.columns:
        if df[column].dtype == "float64" or df[column].dtype == "int64" :
            try:
                _, p_value = lilliefors(df[column],dist='norm', pvalmethod='table')
                normality_results[column] = p_value
            except Exception as e:
                normality_results[column] = str(e)
        else:
            normality_results[column] = "Not numeric type"

    return pd.DataFrame(normality_results.values(), index=normality_results.keys(), columns=['P-value'])


def check_normality_by_plot(df :pd.DataFrame) ->None:

    """
        Check normality of the data by plot.
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            None
    """

    filtered_columns = df.filter(regex=r'^(?!.*position).*$', axis=1)
    filtered_columns = filtered_columns.filter(regex=r'^(?!.*encoded).*$', axis=1)
    numeric_features = filtered_columns.select_dtypes(include=['float64', 'int64']).columns

    # calculate number of plots
    num_plots = len(numeric_features)
    num_rows = (num_plots - 1) // 3 + 1
    num_cols = min(num_plots, 3)

    plt.figure(figsize=(num_cols * 6, num_rows * 4))

    # plot histograms
    for i, feature in enumerate(numeric_features, 1):
        
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(df[feature], kde=True)
        plt.title(feature)

    plt.tight_layout()
    plt.show()
        



def kendall_correlation_matrix(df :pd.DataFrame) -> dict:
    """
        Calculation of the correlation coefficient between all pairs of features.
        Method choose Kendall correlation coefficient because it is more robust to outliers 
        than Pearson correlation coefficient. And data is not always normally distributed.

    """
    numeric_features = df.select_dtypes(include=['float64', 'int64'])
    columns = numeric_features.columns
    correlations = {}
    for col1, col2 in combinations(columns, 2):
        correlation, _ = kendalltau(numeric_features[col1], numeric_features[col2])
        key = f"{col1}_relationship_{col2}"
        correlations[key] = correlation

    sorted_dict = dict(sorted(correlations.items(), key=lambda x: x[1] if not pd.isna(x[1]) else float('inf')))
    return sorted_dict


def correlation_in_plots(correlation_dict: dict, df:pd.DataFrame) -> None:
    '''
        Plot scatter plots for features with high correlation.
    '''

    #create plots for features with high correlation abs()>0.6
    filtered_correlation_dict = {k: v for k, v in correlation_dict.items() if abs(v) > 0.6}
    fig, axes = plt.subplots(nrows=1, ncols=len(filtered_correlation_dict), figsize=(15, 5))

    for i, (key, value) in enumerate(filtered_correlation_dict.items()):
        columns = key.split("_relationship_")
        column1 = columns[0]
        column2 = columns[1]
        
        sns.scatterplot(data=df, x=column1, y=column2, ax=axes[i])
        if i % 2 == 0:  # Чередование заголовков
            axes[i].set_title(key, pad=20,fontsize=10)  # Заголовок сверху
        else:
            axes[i].set_title(key, pad=-20,fontsize=10)  # Заголовок снизу

    plt.tight_layout()
    plt.show()


def warm_matrix_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a correlation matrix for the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The correlation matrix.
    """
    filtered_columns = df.filter(regex=r'^(?!.*position).*$', axis=1)
    filtered_columns = filtered_columns.filter(regex=r'^(?!.*encoded).*$', axis=1)

    numeric_features = filtered_columns.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_features.corr()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    for text in heatmap.texts:
        text.set_fontsize(8)
    plt.show()



#Check distributions and ranges of features (use plots);
check_lilliefors_normality(df)
check_normality_by_plot(df)

#Check relationships between pairs of differents features (use plots and correlation coefficients);

correlation_dict= kendall_correlation_matrix(df) #all correlation coefficients
correlation_in_plots(correlation_dict, df) #plots for high correlation coefficients

#Check relationships between all features (use plots).
warm_matrix_correlation(df)

