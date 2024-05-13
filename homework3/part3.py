import pandas as pd

df = pd.read_csv("homework3/homework3_data_preprocessed_result.csv")

'''
I create only one new feature, which is global_stat. It`s consist information about global
physical and technical characteristics of the player.
Another features are atomic and self-sufficient.
Maybe it can be more useful to divide this feature into two separate features, but I think
it will be too informative.

'''
selected_columns = ['fk_accuracy', 'ball_control', 'sprint_speed', 'agility', 'reactions',
                    'balance', 'shot_power', 'stamina', 'strength', 'aggression', 'interceptions', 'penalties', 'defending']
df['global_stat'] = df[selected_columns].mean(axis=1)

"""
As a result we have dataframe where most of data have numeric view. Only few columns have object type.
Of course this dataframe consists too much features, but its possible to combine them to get specific information.
"""
df.to_csv("homework3/homework3_data_result.csv", index=False)
print(df.iloc[0])