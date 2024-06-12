import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler


def preprocess_data():
    df = pd.read_csv("homework8/homework8_data.csv")
    encoder = OneHotEncoder()
    encoded_sex = encoder.fit_transform(df[['sex']])
    encoded_df_sex = pd.DataFrame(encoded_sex.toarray(), columns=['male', 'female'])
    encoded_pclass = encoder.fit_transform(df[['pclass']])
    encoded_df_pclass = pd.DataFrame(encoded_pclass.toarray(), columns=['pclass1', 'pclass2', 'pclass3'])
    encoded_embarked = encoder.fit_transform(df[['embarked']])
    encoded_df_embarked = pd.DataFrame(encoded_embarked.toarray(), columns=['embarked0', 'embarked1', 'embarked2'])
    encoded_df = pd.concat([df, encoded_df_sex, encoded_df_pclass, encoded_df_embarked], axis=1)
    encoded_df.drop(['sex', 'pclass', 'embarked'], axis=1, inplace=True)
    min_max = MinMaxScaler()
    df_min_max = pd.DataFrame(min_max.fit_transform(encoded_df), columns=encoded_df.columns)
    result_df = pd.DataFrame(df_min_max, columns=encoded_df.columns)
    return result_df

