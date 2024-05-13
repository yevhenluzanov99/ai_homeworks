import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
"""
sybmols that should be removed from the column names
"""
UNWANTED_SYMBOLS = [
    "/",
    "↓",
    "↑",
    "★",
    "°",
    "\n",
]
"""
units of measurement that should be removed from the data
"""
UNITS_MEASUREMENT = [
    "°C",
    "%",
    "mm",
    "kg",
    "cm",
]


def standartize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns according to basic column names standards.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    # Rename columns to lowercase and replace spaces with underscores
    df.rename(columns=lambda x: x.lower(), inplace=True)
    df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    # Remove unwanted symbols from column names
    for column in df.columns:
        if any(symbol in column for symbol in UNWANTED_SYMBOLS):
            new_column_name = "".join(
                [char for char in column if char not in UNWANTED_SYMBOLS]
            )
            df.rename(columns={column: new_column_name}, inplace=True)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data from any unnecessary and unwanted characters.
    Standardize data so it follows commonly accepted data formats, data units, etc.
    Make numerical variables have numerical Pythonic dtypes.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned and standardized DataFrame.
    """
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].apply(
                lambda x: "".join([char for char in x if char not in UNWANTED_SYMBOLS])
            )
    # Remove units of measurement from the data
    for column in df.columns:
        if df[column].dtype == "object":
            for measurement in UNITS_MEASUREMENT:
                df[column] = df[column].apply(lambda x: x.replace(measurement, ""))
    # Convert numerical variables to numerical Pythonic dtypes
    # Convert weight from lbs to kg
    for index, value in df["weight"].items():
        if "lbs" in value:
            weight_lbs = float(value.replace("lbs", ""))
            df.at[index, "weight"] = round(weight_lbs * 0.453592, 2)
    # Convert height from feet and inches to cm
    for index, value in df["height"].items():
        if "'" in value and '"' in value:
            feet, inches = value.split("'")
            height_cm = int(feet) * 30.48 + int(inches.replace('"', "")) * 2.54
            df.at[index, "height"] = round(height_cm, 2)
    # Convert value and wage to numerical values
    for index, value in df["value"].items():
        if "M" in value:
            value_m = float(value.replace("M", "").replace("€", ""))
            df.at[index, "value"] = round(value_m * 1000000, 2)
        elif "K" in value:
            value_k = float(value.replace("K", "").replace("€", ""))
            df.at[index, "value"] = round(value_k * 1000, 2)
        else:
            df.at[index, "value"] = round(float(value.replace("€", "")), 2)

    for index, value in df["wage"].items():
        if "M" in value:
            value_m = float(value.replace("M", "").replace("€", ""))
            df.at[index, "wage"] = round(value_m * 1000000, 2)
        elif "K" in value:
            value_k = float(value.replace("K", "").replace("€", ""))
            df.at[index, "wage"] = round(value_k * 1000, 2)
        else:
            df.at[index, "wage"] = round(float(value.replace("€", "")), 2)
    # Cross check data  
    #if there are any missing not-converted values they will be printed in exception
    #also convert str data to float if possible
    for column in df.columns:
        if df[column].dtype == "object":
            try:
                df[column] = df[column].astype(float)
            except Exception as e:
                #print(e)
                continue
    # Convert date columns to datetime
    df["joined"] = pd.to_datetime(df["joined"])

    return df


def handle_data_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find and handle duplicates, missing values, and outliers if they are present and any handling is appropriate.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed and missing values handled.
    """
    #Check for duplicates and drop them if they are present
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    if duplicate_rows.empty:
        pass
    else:
        df.drop_duplicates(inplace=True)
    #Check for missing values and drop them if they are present
    missing_values = df.isnull().sum()

    for column, count in missing_values.items():
        if count != 0:
            df = df.dropna(subset=[column])
    #Check for outliers. 
    #Create descriptive statistics for the DataFrame and check all columns for outliers value by printing and analyzing them
    df_description = df.describe().copy()
    '''
        Index(['age', 'ova', 'height', 'weight', 'bov', 'joined', 'value', 'wage',
       'fk_accuracy', 'ball_control', 'sprint_speed', 'agility', 'reactions',
       'balance', 'shot_power', 'stamina', 'strength', 'aggression',
       'interceptions', 'penalties', 'defending', 'wf', 'ir'],
      dtype='object')
    '''
    #print(df_description['age'])
    return df


def bin_and_encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data binning, one-hot encoding, or label encoding where appropriate.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with binned and encoded data.
    """
    label_encoder = LabelEncoder()
    #Label encoding for categorical variables
    df["nationality_encoded"] = label_encoder.fit_transform(df["nationality"])
    df["preferred_foot_encoded"] = label_encoder.fit_transform(df["preferred_foot"])
    df["best_position_encoded"] = label_encoder.fit_transform(df["best_position"])
    df["club_encoded"] = label_encoder.fit_transform(df["club"])
    #One-hot encoding for categorical variables
    df_encoded = df["positions"].str.get_dummies(", ")
    df_encoded = df_encoded.add_prefix("position_")
    df_final = pd.concat([df, df_encoded], axis=1)
    #Data binning for numerical variables and label encoding them
    height_bins = [0, 170, 180, 190, float("inf")]
    height_bin_labels = ["<=170", "171-180", "181-190", ">=191"]
    df_final["height_bins"] = pd.cut(
        df_final["height"], bins=height_bins, labels=height_bin_labels
    )
    df_final["height_bins_encoded"] = label_encoder.fit_transform(
        df_final["height_bins"]
    )
    weight_bins = [0, 70, 80, 90, float("inf")]
    weight_bin_labels = ["<=70", "71-80", "81-90", ">=91"]
    df_final["weight_bins"] = pd.cut(
        df_final["weight"], bins=weight_bins, labels=weight_bin_labels
    )
    df_final["weight_bins_encoded"] = label_encoder.fit_transform(
        df_final["weight_bins"]
    )
    age_bins = [0, 20, 25, 30, 35, 40, 45, 50, float("inf")]
    age_bin_labels = [
        "<=20",
        "21-25",
        "26-30",
        "31-35",
        "36-40",
        "41-45",
        "46-50",
        ">=51",
    ]
    df_final["age_bins"] = pd.cut(df_final["age"], bins=age_bins, labels=age_bin_labels)
    df_final["age_bins_encoded"] = label_encoder.fit_transform(df_final["age_bins"])
    return df_final

def get_year(text):
    pattern = r'\b\d{4}\b'  # Регулярное выражение для поиска четырех цифр (\d{4}) в слове (\b - граница слова)
    match = re.search(pattern, text)
    if match:
        return match.group()  # Возвращаем найденное значение
    else:
        return 0


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features where appropriate.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with scaled features.
    """
    #Scale date contract column for contract_from and contract_to for better analysis
    df[["contract_from", "contract_to"]] = df["contract"].str.split(" ~ ", expand=True)
    
    for index, value in df["contract"].items():
        if " ~ " not in value:
            # if contract has active status we set contract_to to 2100
            df.at[index, "contract_from"] = get_year(value)
            df.at[index, "contract_to"] = 2100

    df["contract_from"] = df["contract_from"].astype('int64')
    df["contract_to"] = df["contract_to"].astype('int64')
    return df


df = pd.read_csv("homework3/homework3_data.csv")
#Rename columns according to basic column names standards;
df = standartize_columns(df)
#Clean data from any unnessesary and unwanted characters;
#Standardize data so it follows commonly accepted data formats, data units and etc;
#Make numerical variables to have numerical Pythonic dtypes;
df = clean_data(df)
#Find and handle duplicates, missing values and outliers if they are present and any handling is appropriate;
df = handle_data_quality_issues(df)
#Perform data binning, one-hot encoding or label encoding where it is appropriate;
df = bin_and_encode_data(df)
#Scale features where it is appropriate.
df = scale_features(df)
#Save the final DataFrame to a CSV file
df.to_csv("homework3/homework3_data_preprocessed_result.csv", index=False)
