
# Datasets

    homework3_data.csv - Input dataset.
    homework3_data_preprocessed_result.csv - dataset after Data Preprocessing part .
    homework3_data_result.csv - final dataset after all steps.

# Possible mistakes and question

    in part 2 when i try to understand data in dataset, i create  function check_normality_by_plot to see which distribution 
    have all features. And some plots show me that data has normal distribution. But after this i calculate lilliefors ratio
    and it show that all data features have not-normal distribution. After this i try to make data scalling by two methods:
    Log-Transformation:
        df[column] = np.log1p(df[column])
    Ranking:
        df[column] = df[column].rank()

    And olny in ranking method i found some segments with not-normal distribution. But Log-Transformation didn`t show me this 
    segments. So i`m not shure about correctness