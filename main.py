import pandas
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from colorama import Fore
import re
import os
import logging
from datetime import datetime
import traceback
import easygui

#import numpy as np

pandas.options.display.float_format = '{:,.2f}'.format

print('Intitializing...')

#file_path = 'data/horford_reg_szn.csv'
#file_path = 'data/t_haliburton_23-24_regszn.csv'
file_path = easygui.fileopenbox()
#file_path_t_haliburton_playoffs = 'data/t_haliburton_23-24_playoffs.csv'

settings = {
    #default values
    'dataset_path': 'data/t_haliburton_23-24_regszn.csv',
    'test_size': 0.2,
    'random_state': 42
}


#set1 = pandas.read_csv(file_path_t_haliburton_regszn, parse_dates=['Date'], dtype={'G': str})
#set2 = pandas.read_csv(file_path_t_haliburton_playoffs, parse_dates=['Date'], dtype={'G': str})
os.system('cls')

def convert_mp_to_minutes(mp_str):
    try:
        if isinstance(mp_str, pandas.Timestamp):
            time_str = mp_str.strftime('%H:%M:%S')
        else:
            time_str = mp_str

        # Split the time string into hours, minutes, and seconds
        time_parts = time_str.split(':')
        minutes = int(time_parts[0])
        seconds = int(time_parts[1])
        # Calculate total minutes
        total_minutes = float(minutes + seconds/60)
        return total_minutes
    except Exception as e:
        print(f"Error converting {mp_str}: {e}")
        return 0
    
def calculate_trend(series, window=5):
    trend = []
    for i in range(len(series)):
        if i < window - 1:
            trend.append(np.nan)
        else:
            y = series[i - window + 1:i + 1]
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            trend.append(slope)
    return pandas.Series(trend, index=series.index)

def preprocess(file_path):
    
    #read csv
    df = pandas.read_csv(file_path, parse_dates=['Date'], dtype={'MP': str})

    #split the result and point diff from the W/L column
    df[['Result', 'Point_Diff']] = df['W/L'].str.extract(r'([WL])\s*\(([-+]?\d+)\)?')

    #drop NaNs from the MP because the model freaks out if there's any NaNs
    df = df.dropna(subset=['MP'])

    #drop extraneous/old columns
    df = df.drop(columns=['Rk', 'G', 'Age', 'Tm', 'W/L'])

    # convert 'Point_Diff' to numeric (if applicable)
    df['Point_Diff'] = pandas.to_numeric(df['Point_Diff'], errors='coerce')

    #store encoded data in a dictionary for user reference
    label_encoders = {}
    label_encoders['Opp'] = LabelEncoder()
    label_encoders['Result'] = LabelEncoder()
    label_encoders['LOC'] = LabelEncoder()
    
    #encode data that must be in boolean format (W/L must be 0 or 1, etc.) 
    df['LOC_encoded'] = label_encoders['LOC'].fit_transform(df['LOC'])
    df['Opp_encoded'] = label_encoders['Opp'].fit_transform(df['Opp'])
    df['Result_enc'] = label_encoders['Result'].fit_transform(df['Result'])

    #convert minutes played to a number and strip colons
    df['MP'] = df['MP'].apply(convert_mp_to_minutes)
    df['MP'] = df['MP'].astype(float)

    # Convert to seconds since epoch. This one is less of a requirement for the model and more of one
    # for the model just to work as it doesn't do well with datetime objects. 
    if 'Date' in df.columns:
        df['Date'] = pandas.to_datetime(df['Date']).astype('int64') / 10**9  

    #remove any NaNs if still exist
    df.fillna(0, inplace=True)

    #convert to numeric
    df['MP'] = pandas.to_numeric(df['MP'], errors='coerce')
    df['FG%'] = pandas.to_numeric(df['FG%'], errors='coerce')
    df['3P%'] = pandas.to_numeric(df['3P%'], errors='coerce')

    #feature generation
    df['above_line'] = df['PTS'] > df['Line']
    df['above_line'] = df['above_line'].astype('int')
    df['rebounds_assists_ratio'] = df['TRB'] / df['AST']
    df['pts_reb+ast_ratio'] = df['PTS'] / (df['TRB'] + df['AST'])
    df['3pa_fga_ratio'] = df ['3PA'] / (df['FGA'] - -df['3PA'])
    df.replace([float('inf'), -float('inf')], 0, inplace=True)

    # Calculate season average stats
    season_avg_pts = df['PTS'].mean()
    season_avg_ast = df['AST'].mean()
    season_avg_trb = df['TRB'].mean()

    # Calculate performance deviation from average
    df['deviation_pts'] = df['PTS'] - season_avg_pts
    df['deviation_ast'] = df['AST'] - season_avg_ast
    df['deviation_trb'] = df['TRB'] - season_avg_trb

    # Calculate rolling statistics with min_periods=1 to avoid NaN for fewer games
    df['rolling_std_pts'] = df['PTS'].rolling(window=5, min_periods=1).std()
    df['rolling_std_ast'] = df['AST'].rolling(window=5, min_periods=1).std()
    df['rolling_std_trb'] = df['TRB'].rolling(window=5, min_periods=1).std()

    df['rolling_mean_pts'] = df['PTS'].rolling(window=5, min_periods=1).mean()
    df['rolling_mean_ast'] = df['AST'].rolling(window=5, min_periods=1).mean()
    df['rolling_mean_trb'] = df['TRB'].rolling(window=5, min_periods=1).mean()

    df['z_score_pts'] = (df['PTS'] - df['rolling_mean_pts']) / df['rolling_std_pts'].replace(0, 1)
    df['z_score_ast'] = (df['AST'] - df['rolling_mean_ast']) / df['rolling_std_ast'].replace(0, 1)
    df['z_score_trb'] = (df['TRB'] - df['rolling_mean_trb']) / df['rolling_std_trb'].replace(0, 1)

    df['rolling_mp'] = df['MP'].rolling(window=5, min_periods=1).mean()

    # Calculate trends
    df['trend_mp'] = calculate_trend(df['MP'])
    df['trend_pts'] = calculate_trend(df['PTS'])
    df['trend_ast'] = calculate_trend(df['AST'])
    df['trend_trb'] = calculate_trend(df['TRB'])


    df.fillna(0, inplace=True)

    return df, label_encoders

    return df, label_encoders


    return df, label_encoders

def display_label_encodings(label_encoder):
    os.system('cls')
    for feature, encoder in label_encoder.items():
        print(Fore.LIGHTCYAN_EX + f"Label encodings for {feature}:")
        for index, label in enumerate(encoder.classes_):
            print(Fore.WHITE + f"{index}: {label}")
        print()
    input('\nPress any key to continue.')
    os.system('cls')
    
        
def print_settings():
    print("\nCurrent Settings:")
    for key, value in settings.items():
        print(f"{key}: {value}")
    print()

def change_setting():
    os.system('cls')
    print_settings()
    setting_key = input("Enter the setting you want to change (or 'x' to return): ")
    if setting_key.lower() == 'x':
        return
    if setting_key in settings:
        new_value = input(f"Enter new value for {setting_key} (current: {settings[setting_key]}): ")
        # Additional validation based on setting type can be added here
        if setting_key in ['test_size', 'random_state']:
            new_value = float(new_value) if setting_key == 'test_size' else int(new_value)
        settings[setting_key] = new_value
        print(f"{setting_key} updated to {new_value}.")
    else:
        print("Invalid setting key.")

def settings_menu():
    while True:
        os.system('cls')
        print(Fore.LIGHTBLUE_EX + "Settings Menu" + Fore.WHITE)
        print_settings()
        print("1 - Change setting")
        print("2 - Exit settings")
        choice = input('\n')
        if choice == '1':
            change_setting()
        elif choice == '2':
            os.system('cls')
            break
        else:
            print("Invalid choice, please try again.")

def setup_logging(): 
    timestamp = datetime.now().strftime('%m-%d-%Y')
    log_dir = f'logs/{timestamp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    error_log_path = os.path.join(log_dir, 'errors.log')
    error_logger = logging.getLogger('error_logger')
    error_handler = logging.FileHandler(error_log_path)
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    error_handler.setFormatter(error_formatter)
    error_logger.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)
    
    prediction_log_path = os.path.join(log_dir, 'predictions.log')
    prediction_logger = logging.getLogger('prediction_logger')
    prediction_handler = logging.FileHandler(prediction_log_path)
    prediction_formatter = logging.Formatter('%(asctime)s - %(message)s')
    prediction_handler.setFormatter(prediction_formatter)
    prediction_logger.setLevel(logging.INFO)
    prediction_logger.addHandler(prediction_handler)
    
    return error_logger, prediction_logger



print(Fore.YELLOW + 'Welcome to the ML Basketball Tool!\nPress enter to get started!')
while True:
    user_input = input()
    if(user_input == ''):
        break

#initialize preprocessing
try:
    set1, mappings = preprocess(file_path)
    
except Exception as e:
    print(Fore.RED + f'Error during preprocessing: {e}')
print(Fore.GREEN + f'Preprocessing complete of file: {file_path}\n')

#initialize logging service
try:
    error_logger, prediction_logger = setup_logging()
except Exception as e:
    print(Fore.RED + f'Error during intializing logging service: {e}')
print(Fore.GREEN + f'Logging service setup complete. See file path /logs/{datetime.now().strftime('%m-%d-%Y')} for output logs')

input(Fore.WHITE + '\nPress any key to continue.')



#diagram service main loop
def diagram_service():
    while(True):
       
        print(Fore.LIGHTGREEN_EX + 'VISUALIZATIONS DASHBOARD:\nChoose from the following numbered options\n')
        data = input(Fore.WHITE + "1 - Points Histogram\n2 - Scatter plot of points vs. minutes played\n3 - Scatter plot of points vs. rebounds_assists_ratio\n4 - Scattor plot of points vs. assists\n5 - Show entire data table\nx - Back to previous screen\n")
        match data:
            case '1':
                # Histogram of points scored
                plt.hist(set1['PTS'], bins=20, edgecolor='black')
                plt.xlabel('Points Scored')
                plt.ylabel('Frequency')
                plt.title('Distribution of Points Scored')
                plt.show()
            case '2':
                # Scatter plot of points vs. minutes played
                plt.scatter(set1['MP'], set1['PTS'])
                plt.xlabel('Minutes Played')
                plt.ylabel('Points Scored')
                plt.title('Points vs. Minutes Played')
                plt.show()
            case '3':
                # Scatter plot of points vs. rebounds_assists_ratio
                plt.scatter(set1['rebounds_assists_ratio'], set1['PTS'])
                plt.xlabel('Rebounds to Assists Ratio')
                plt.ylabel('Points Scored')
                plt.title('Points vs. Rebounds to Assists Ratio')
                plt.show()
            case '4':
                # Scattor plot of points vs. assists
                plt.scatter(set1['AST'], set1['PTS'])
                plt.xlabel('Assists Recorded')
                plt.ylabel('Points Scored')
                plt.title('Points vs. Assists')
                plt.show()
            case '5':
                os.system('cls')
                display_set1 = set1.copy()
                display_set1['Date'] = pandas.to_datetime(display_set1['Date'], unit='s')
                print(display_set1)
                input('\nPress any key to continue.')
                os.system('cls')
            case 'x':
                os.system('cls')
                break
        os.system('cls')

#ANSI cleaner
def ansi_cleaner(text):
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', text)

avg_pts = set1['PTS'].mean()

def train_model():
    try:
        # model creation
        X = set1[['Point_Diff', 'Result_enc', 'LOC_encoded', 'Opp_encoded', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'pts_reb+ast_ratio', '3pa_fga_ratio', 'PTS', 'rolling_std_pts', 'rolling_std_ast', 'rolling_std_trb', 'z_score_pts', 'z_score_ast', 'z_score_trb', 'rolling_mp', 'trend_mp', 'trend_pts', 'trend_ast', 'trend_trb']]
        #X = set1.drop(columns=['Line'])
        y = set1['above_line'].astype(int)

        os.system('cls')

        #create train and test objects from the train_test_split method according to settings paramters
        #default test_size is 0.2, meaning %20 of the data is reserved for test cases and the rest for training data
        #random state is the seed for the random number gen
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings['test_size'], random_state=settings['random_state'])
    
        #create and fit the model according to the generated training data
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        #predict against the test cases to generate evaluation data.
        y_pred = model.predict(X_test)

        return model, X_train, X_test, y_train, y_test, y_pred, X, y
    except Exception as e:
        print(Fore.RED + f"Error during model training: {e}")
        error_logger.error(ansi_cleaner(f"An error occurred: {str(e)}"))
    print(Fore.LIGHTGREEN_EX + 'Data trained!')

model, X_train, X_test, y_train, y_test, y_pred, X, y  = train_model()

#test data
# new_data = pandas.DataFrame({
#     'MP': [set1.iloc[:6]['MP'].mean()],
#     'FG%': [set1.iloc[:6]['FG%'].mean()],
#     '3P%': [set1.iloc[:6]['3P%'].mean()],
#     'FT%': [set1.iloc[:6]['FT%'].mean()],
#     'TRB': [set1.iloc[:6]['TRB'].mean()],
#     'AST': [set1.iloc[:6]['AST'].mean()],
#     'STL': [set1.iloc[:6]['STL'].mean()],
#     'BLK': [set1.iloc[:6]['BLK'].mean()],
#     'TOV': [set1.iloc[:6]['TOV'].mean()],
#     'rebounds_assists_ratio': [set1.iloc[:6]['TRB'].mean() / set1.iloc[:6]['AST'].mean()],
#     'pts_reb+ast_ratio': [set1.iloc[:6]['PTS'].mean() / (set1.iloc[:6]['TRB'].mean() + set1.iloc[:6]['AST'].mean())],
#     '3pa_fga_ratio': [set1.iloc[:6]['3PA'].mean() / (set1.iloc[:6]['FGA'].mean() - set1.iloc[:6]['3PA'].mean())],
#     'PTS': [set1.iloc[:6]['PTS'].mean()],
#     'Line': [18.5],
#     'Opp_encoded' : [1],
#     'LOC_encoded': [1],
#     'Result_enc': [0],
#     'Point_Diff': [-9]

# })

#new_data = pandas.DataFrame()

#print(set1.iloc[:6]['MP'].mean())

#INTERACTIVE QUERIES
#MAIN LOOP FOR MODEL ANALYSIS
while(True):
    print(Fore.GREEN + 'Choose from the following options:\n')
    data = input(Fore.WHITE + '1 - Visualizations Dashboard\n2 - Accuracy/Classification Report for backtested data\n3 - Re-train the model\n4 - Enter custom query to predict a future match\n5 - Settings\n6 - Label Mappings\nx - Quit\n')
    os.system('cls')
    match data:
        case '1':
            diagram_service()
        case '2':
            # evaluate
            os.system('cls')
            print(Fore.YELLOW + f'This report was generated with a test_size of {settings["test_size"]}, meaning %{str(settings["test_size"]).split('.')[1] + '0'} of the data was reserved for test cases and the rest is used for training.\n')
            accuracy = accuracy_score(y_test, y_pred)
            print(Fore.WHITE + f'Accuracy: {accuracy:.2f}')

            print('Classification Report:')
            print(classification_report(y_test, y_pred))

            print('Confusion Matrix:')
            print(confusion_matrix(y_test, y_pred))
            data = input('\nPress any key to exit!')
            os.system('cls')
        case '3':
            os.system('cls')
            model, X_train, X_test, y_train, y_test, y_pred, X, y = train_model()
            print(Fore.YELLOW + 'Model re-trained')
            print(Fore.WHITE)
        case '4': #custom query menu
            while(True):
                print('Enter a future match\'s implied numbers. If no specific implied value is known, enter \'avg\' for the value and the algorithm will use the previous 6-match average. \nSee label mappings/documentation for official list of codes and explanations.')
                #print('FORMAT: [Minutes Played], [Field Goal %], [3 Pointer %], [Free Throw %], [Rebounds], [Assists], \n[Steals], [Blocks], [Turnovers], [Points], [Opponent Code], [Home/Away (0 is home, 1 is away)], [Expected Win/Loss (0 L/ 1 W)], [Spread], [Expected over/under (0 Under/1 Over)]')
                print("FORMAT: 'Point_Diff', 'Result_enc', 'LOC_encoded', 'Opp_encoded', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'pts_reb+ast_ratio', '3pa_fga_ratio', 'PTS'")
                print('EXAMPLE: +6.5, 0, 0, 1, avg, avg, avg, avg, 4.5, 7.5, 0.5, 0.5, 3.5, avg, avg, avg, 18.5')
                print('\n')
                print('Enter your stats (or \'x\' to return to previous menu):')
                data = input().split(',')
                if data[0] == 'x':
                    os.system('cls')
                    break

                for i in range(0, len(data)):
                    tmp = data[i].strip()
                    data[i] = tmp

                try:
                    new_data = pandas.DataFrame()
                    field_list = ['Point_Diff', 'Result_enc', 'LOC_encoded', 'Opp_encoded', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'pts_reb+ast_ratio', '3pa_fga_ratio', 'PTS']
                    #field_list = X.columns
                    for i in range(0, len(data)):
                        if(data[i] == 'avg'):
                            new_data[field_list[i]] = set1.iloc[:6][field_list[i]].mean()
                            print(new_data[field_list[i]])
                        else:
                            new_data[field_list[i]] = [data[i].strip('+')]
                    
                    new_data['rebounds_assists_ratio'] = [set1.iloc[:6]['TRB'].mean() / set1.iloc[:6]['AST'].mean()]
                    new_data['pts_reb+ast_ratio'] = [set1.iloc[:6]['PTS'].mean() / (set1.iloc[:6]['TRB'].mean() + set1.iloc[:6]['AST'].mean())]
                    new_data['3pa_fga_ratio'] = [set1.iloc[:6]['3PA'].mean() / (set1.iloc[:6]['FGA'].mean() - set1.iloc[:6]['3PA'].mean())]

                    new_data['PTS'] = new_data['PTS'].astype(float)
                    new_data['AST'] = new_data['AST'].astype(float)
                    new_data['TRB'] = new_data['TRB'].astype(float)

                    # Calculate rolling standard deviation for points, assists, and rebounds over the last 5 games
                    rolling_window = min(5, len(set1))
                    new_data['rolling_std_pts'] = set1['PTS'].rolling(window=5, min_periods=1).std().iloc[-1]
                    new_data['rolling_std_ast'] = set1['AST'].rolling(window=5, min_periods=1).std().iloc[-1]
                    new_data['rolling_std_trb'] = set1['TRB'].rolling(window=5, min_periods=1).std().iloc[-1]

                    # Calculate rolling mean and standard deviation for points
                    rolling_mean_pts = set1['PTS'].rolling(window=5, min_periods=1).mean().iloc[-1]
                    rolling_std_pts = set1['PTS'].rolling(window=5, min_periods=1).std().iloc[-1]

                    # Calculate Z-score for points
                    new_data['z_score_pts'] = (new_data['PTS'] - rolling_mean_pts) / rolling_std_pts if rolling_std_pts != 0 else 0

                    # Repeat for assists and rebounds
                    rolling_mean_ast = set1['AST'].rolling(window=5, min_periods=1).mean().iloc[-1]
                    rolling_std_ast = set1['AST'].rolling(window=5, min_periods=1).std().iloc[-1]
                    new_data['z_score_ast'] = (new_data['AST'] - rolling_mean_ast) / rolling_std_ast if rolling_std_ast != 0 else 0

                    rolling_mean_trb = set1['TRB'].rolling(window=5, min_periods=1).mean().iloc[-1]
                    rolling_std_trb = set1['TRB'].rolling(window=5, min_periods=1).std().iloc[-1]
                    new_data['z_score_trb'] = (new_data['TRB'] - rolling_mean_trb) / rolling_std_trb if rolling_std_trb != 0 else 0

                    new_data['rolling_mp'] = new_data['MP'].rolling(window=5, min_periods=1).mean()

                    new_data['trend_mp'] = set1['MP']
                    new_data['trend_pts'] = set1['PTS']
                    new_data['trend_ast'] = set1['AST']
                    new_data['trend_trb'] = set1['TRB']

                    new_data.fillna(0, inplace=True)

                    os.system('cls')
                    
                    next_game_prediction = model.predict(new_data)

                    #print(np.mean(y_pred == y_test))
                    your_data = Fore.LIGHTYELLOW_EX + f'YOUR DATA:' + Fore.WHITE + f'\n{new_data}\n'
                    prediction_output = Fore.YELLOW + f'NEXT GAME:' + Fore.WHITE + f'\nPrediction: {Fore.GREEN + "Above Line" if next_game_prediction[0] else Fore.LIGHTRED_EX + "Below Line"}'
                    prediction_logger.info(ansi_cleaner(f'Prediction Result:\n{your_data}\n{prediction_output}\n\n'))

                    print(your_data)
                    print(prediction_output)

                    input(Fore.WHITE + '\nPress any key to continue.')
                    os.system('cls')

                except Exception as e:
                    #print(f'\n{X[X.isna().any(axis=1)]}')
                    print(traceback.format_exc())

                    print(f'Error during data input: {e}\nYour data was likely input incorrectly; Please follow the format.\nPress any key to retry.')
                    error_logger.error(ansi_cleaner(f"An error occurred: {str(e)}"))
                    data = input()
                    os.system('cls')

        case '5':
            settings_menu()
        case '6':
            display_label_encodings(mappings)
        case 'x':
            quit()


print(Fore.WHITE)
quit()
#new_data = set1.iloc[:6]

#first_row = set2.head(1).copy()
#print(set2.head(1))
#ORIGINAL MODEL
next_game_prediction = model.predict(new_data[X.columns])

#print(np.mean(y_pred == y_test))

print(f'NEXT GAME: \nPrediction: {"Above Line" if next_game_prediction[0] else "Below Line"}')

print(new_data)
#print(y_pred)
#print(y_test)
quit()
