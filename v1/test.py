import rc_util
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from colorama import Fore
import basketball_reference_scraper as brs

def_rankings = {
    "OKC": 1, "HOU": 2, "ORL": 3, "MEM": 4, "GSW": 5, "LAC": 6, "BOS": 7, "DAL": 8, "CLE": 9, "MIA": 10,
    "SAS": 11, "DET": 12, "POR": 13, "MIN": 14, "MIL": 15, "SAC": 16, "CHA": 17, "PHI": 18, "DEN": 19,
    "ATL": 20, "TOR": 21, "IND": 22, "NYK": 23, "PHX": 24, "BKN": 25, "LAL": 26, "NOP": 27, "CHI": 28,
    "UTA": 29, "WAS": 30
}


os.system('cls')

print(Fore.LIGHTYELLOW_EX +'Basketball Stats v2' )
input('Press any key to continue.' + Fore.WHITE)

os.system('cls')

roster = rc_util.get_roster(input( Fore.LIGHTBLUE_EX + 'Enter a team code: '+ Fore.WHITE), 10)

os.system('cls')

player = rc_util.roster_picker(roster)

stats = rc_util.get_stats_v2(player[1])
stats = stats[~stats['GS'].isin(['Inactive', 'Did Not Dress', 'GS'])]
stats[['Result', 'Spread']] = stats['Unnamed: 7'].str.extract(r'([WL])\s*\(([-+]?\d+)\)')
stats['Spread'] = stats['Spread'].apply(lambda x: x.replace('+', ''))
stats['MP'] = stats['MP'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)

#stats['Opp_code'] = stats['Opp'].astype('category').cat.codes



stats['LOC'] = stats['Unnamed: 5'].apply(lambda x: 1 if x == '@' else 0)
stats['MP_avg_5'] = stats['MP'].rolling(window=5, min_periods=1).mean()
stats['TRB_avg_5'] = stats['TRB'].rolling(window=5, min_periods=1).mean()
stats['AST_avg_5'] = stats['AST'].rolling(window=5, min_periods=1).mean()


X = stats[['Opp_code', 'LOC', 'Spread', 'GS', 'MP', 'TRB', 'AST', 'MP_avg_5', 'TRB_avg_5', 'AST_avg_5']]
y = stats['PTS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



print(Fore.LIGHTYELLOW_EX + "Your Prediction: " + Fore.WHITE + model.predict())

