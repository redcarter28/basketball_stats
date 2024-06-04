import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from colorama import Fore
import os
#import numpy as np

pandas.options.display.float_format = '{:,.2f}'.format

file_path_t_haliburton_regszn = 'data/t_haliburton_23-24_regszn.csv'
file_path_t_haliburton_playoffs = 'data/t_haliburton_23-24_playoffs.csv'

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
        total_minutes = minutes + seconds/60
        return total_minutes
    except Exception as e:
        print(f"Error converting {mp_str}: {e}")
        return 0
    
def preprocess(file_path):
    
    df = pandas.read_csv(file_path, parse_dates=['Date'], dtype={'MP': str})

    df[['Result', 'Point_Diff']] = df['W/L'].str.extract(r'([WL])\s*\(([-+]?\d+)\)?')

    df = df.dropna(subset=['MP'])

    df = df.drop(columns=['Rk', 'G', 'Age', 'Tm', 'W/L'])

    # Convert 'Point_Diff' to numeric (if applicable)
    df['Point_Diff'] = pandas.to_numeric(df['Point_Diff'], errors='coerce')

    df = pandas.get_dummies(df, columns=['LOC'])

    label_encoders = {}
    label_encoders['Opp'] = LabelEncoder()
    label_encoders['Result'] = LabelEncoder()

    df['Opp_encoded'] = label_encoders['Opp'].fit_transform(df['Opp'])
    df['Result_enc'] = label_encoders['Result'].fit_transform(df['Result'])

    df['MP'] = df['MP'].apply(convert_mp_to_minutes)

    if 'Date' in df.columns:
        df['Date'] = pandas.to_datetime(df['Date']).astype('int64') / 10**9  # Convert to seconds since epoch

    df.fillna(0, inplace=True)

    avg_pts = df['PTS'].mean()

    df['above_line'] = df['PTS'] > df['Line']
    df['rebounds_assists_ratio'] = df['TRB'] / df['AST']
    df['pts_reb+ast_ratio'] = df['PTS'] / (df['TRB'] + df['AST'])
    df['3pa_fga_ratio'] = df ['3PA'] / (df['FGA'] - -df['3PA'])
    df.replace([float('inf'), -float('inf')], 0, inplace=True)

    return df, label_encoders

def display_label_encodings(label_encoder):
    for feature, encoder in label_encoder.items():
        mappings = {index: label for index, label in enumerate(encoder.classes_)}
        print(f"Label encodings for {feature}: {mappings}\n")

print(Fore.YELLOW + 'Welcome to the ML Basketball Tool!\nPress enter to get started!')
while True:
    user_input = input()
    if(user_input == ''):
        break


try:
    set1, mappings = preprocess(file_path_t_haliburton_regszn)
    
except Exception as e:
    print(Fore.RED + f'Error during preprocessing: {e}')

os.system('cls')

#set1 = preprocess(file_path_t_haliburton_playoffs)

#target = set1.iloc[:1]
#set1 = set1.iloc[1:]

#MAIN LOOP FOR APPLICATION
while(True):
    print(Fore.GREEN + f'Preprocessing complete of file: {file_path_t_haliburton_regszn}\n')
    print('Choose from the following numbered options\n')
    data = input(Fore.WHITE + "1 - Points Histogram\n2 - Scatter plot of points vs. minutes played\n3 - Scatter plot of points vs. rebounds_assists_ratio\n4 - Scattor plot of points vs. assists\n5 - Train and run the ML algorithm!\n6 - Show encoded mappings\n7 - Exit\n")
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
            break
        case '6':
            print('Displaying encoded mappings:\n')
            display_label_encodings(mappings)
        case '7':
            quit()
    os.system('cls')


avg_pts = set1['PTS'].mean()

try:
    # model creation
    X = set1[['Point_Diff', 'Result_enc', 'LOC_@', 'Opp_encoded', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'pts_reb+ast_ratio', '3pa_fga_ratio', 'PTS']]
    #X = set1.drop(columns=['Line'])
    y = set1['above_line'].astype(int)

    os.system('cls')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_train = X.iloc[:75]
    #y_train = y.iloc[:75]
    #X_test = X.iloc[75:83]
    #y_test = y.iloc[75:83]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
except Exception as e:
    print(Fore.RED + f"Error during model training: {e}")
print(Fore.LIGHTGREEN_EX + 'Data trained!')

#test data
new_data = pandas.DataFrame({
    'MP': [set1.iloc[:6]['MP'].mean()],
    'FG%': [set1.iloc[:6]['FG%'].mean()],
    '3P%': [set1.iloc[:6]['3P%'].mean()],
    'FT%': [set1.iloc[:6]['FT%'].mean()],
    'TRB': [set1.iloc[:6]['TRB'].mean()],
    'AST': [set1.iloc[:6]['AST'].mean()],
    'STL': [set1.iloc[:6]['STL'].mean()],
    'BLK': [set1.iloc[:6]['BLK'].mean()],
    'TOV': [set1.iloc[:6]['TOV'].mean()],
    'rebounds_assists_ratio': [set1.iloc[:6]['TRB'].mean() / set1.iloc[:6]['AST'].mean()],
    'pts_reb+ast_ratio': [set1.iloc[:6]['PTS'].mean() / (set1.iloc[:6]['TRB'].mean() + set1.iloc[:6]['AST'].mean())],
    '3pa_fga_ratio': [set1.iloc[:6]['3PA'].mean() / (set1.iloc[:6]['FGA'].mean() - set1.iloc[:6]['3PA'].mean())],
    'PTS': [set1.iloc[:6]['PTS'].mean()],
    'Line': [18.5],
    'Opp_encoded' : [1],
    'LOC_@': [1],
    'Result_enc': [0],
    'Point_Diff': [-9]

})

#INTERACTIVE QUERIES
#MAIN LOOP FOR MODEL ANALYSIS
while(True):
    data = input(Fore.WHITE + 'Choose from the following options:\n1 - Accuracy/Classification Report for backtested data\n2 - Enter custom query to predict a future match\n3 - Quit\n')
    os.system('cls')
    match data:
        case '1':
            # evaluate
            print('This report was generated with a test_size of 0.2, meaning %20 of the data was reserved for test cases and the rest is used for training.\n')
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.2f}')

            print('Classification Report:')
            print(classification_report(y_test, y_pred))

            print('Confusion Matrix:')
            print(confusion_matrix(y_test, y_pred))
            data = input('\nPress any key to exit!')
            os.system('cls')
        case '2':
            while(True):
                print('Enter a future match\'s implied numbers. If no specific implied value is known, enter \'avg\' for the value and the algorithm will use the previous 6-match average. \nSee documentation for official list of codes and explanations.')
                print('FORMAT: [Minutes Played], [Field Goal %], [3 Pointer %], [Free Throw %], [Rebounds], [Assists], \n[Steals], [Blocks], [Turnovers], [Points], [Opponent Code], [Home/Away], [Expected Win/Loss], [Spread]')
                print('EXAMPLE: 30, 0.52, 0.37, 0.76, 5, 7, 1, 0, 3, 18.5, 1, 0, 0, +7.5')
                print('\n')
                print('Enter your stats:')
                data = input().split(',')

                for i in range(0, len(data)):
                    tmp = data[i].strip()
                    data[i] = tmp

                try:
                    new_data = pandas.DataFrame()
                    field_list = ['MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'Opp_encoded', 'LOC_@', 'Result_enc', 'Point_Diff']
                    for i in range(0, len(data)):
                        if(data[i] == 'avg'):
                            new_data[field_list[i]] = set1.iloc[:6][field_list[i]].mean()
                        else:
                            new_data[field_list[i]] = [data[i].strip('+')]
                    
                    new_data['rebounds_assists_ratio'] = [set1.iloc[:6]['TRB'].mean() / set1.iloc[:6]['AST'].mean()]
                    new_data['pts_reb+ast_ratio'] = [set1.iloc[:6]['PTS'].mean() / (set1.iloc[:6]['TRB'].mean() + set1.iloc[:6]['AST'].mean())]
                    new_data['3pa_fga_ratio'] = [set1.iloc[:6]['3PA'].mean() / (set1.iloc[:6]['FGA'].mean() - set1.iloc[:6]['3PA'].mean())]


                    # Convert 'Point_Diff' to numeric (if applicable)
                    new_data['Point_Diff'] = pandas.to_numeric(new_data['Point_Diff'], errors='coerce')

        

                    os.system('cls')
                    print(f'YOUR DATA: \n{new_data}')
                    next_game_prediction = model.predict(new_data[X.columns])

                    #print(np.mean(y_pred == y_test))

                    print(f'NEXT GAME: \nPrediction: {"Above Line" if next_game_prediction[0] else "Below Line"}')
                    break

                except Exception as e:
                    print(f'Error during data input: {e}\nYour data was likely input incorrectly; Please follow the format.\nPress any key to retry.')
                    data = input()

                
            break
                

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
