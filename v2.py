import v1.rc_util as rc_util
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from colorama import Fore
import basketball_reference_scraper as brs
from colorist import rgb
from rich.progress import track
import time
from bidict import bidict

os.system('cls')
rgb('Basketball Matchup Tool v1', 255, 191, 0)
print('Initializing...')
#bd = rc_util.get_matchup(rc_util.get_schedule())
bd = bidict()
teams = pd.read_excel('data\\tm_data\\nbacom_tmdata.xlsx')
print('Done!')
time.sleep(1)
os.system('cls')
print('Choose a team:')
print(teams)
team = input()

