import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
#import numpy as np

pandas.options.display.float_format = '{:,.2f}'.format

file_path_t_haliburton_regszn = 'data/t_haliburton_23-24_regszn.csv'
file_path_t_haliburton_playoffs = 'data/t_haliburton_23-24_playoffs.csv'

#set1 = pandas.read_csv(file_path_t_haliburton_regszn, parse_dates=['Date'], dtype={'G': str})
#set2 = pandas.read_csv(file_path_t_haliburton_playoffs, parse_dates=['Date'], dtype={'G': str})

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

    label_encoder = LabelEncoder()
    df['Opp_encoded'] = label_encoder.fit_transform(df['Opp'])
    df['Result_enc'] = label_encoder.fit_transform(df['Result'])

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

    return df

set1 = preprocess(file_path_t_haliburton_regszn)
#set1 = preprocess(file_path_t_haliburton_playoffs)

target = set1.iloc[:1]
#set1 = set1.iloc[1:]

avg_pts = set1['PTS'].mean()
print(set1)
print(avg_pts)

# Histogram of points scored
plt.hist(set1['PTS'], bins=20, edgecolor='black')
plt.xlabel('Points Scored')
plt.ylabel('Frequency')
plt.title('Distribution of Points Scored')
plt.show()

# Scatter plot of points vs. minutes played
plt.scatter(set1['MP'], set1['PTS'])
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.title('Points vs. Minutes Played')
plt.show()

# Scatter plot of points vs. rebounds_assists_ratio
plt.scatter(set1['rebounds_assists_ratio'], set1['PTS'])
plt.xlabel('Rebounds to Assists Ratio')
plt.ylabel('Points Scored')
plt.title('Points vs. Rebounds to Assists Ratio')
plt.show()

# Scattor plot of points vs. assists
plt.scatter(set1['AST'], set1['PTS'])
plt.xlabel('Assists Recorded')
plt.ylabel('Points Scored')
plt.title('Points vs. Assists')
plt.show()

# model
X = set1[['Point_Diff', 'Result_enc', 'LOC_@', 'Opp_encoded', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'rebounds_assists_ratio', 'pts_reb+ast_ratio', '3pa_fga_ratio', 'PTS']]
#X = set1.drop(columns=['Line'])
y = set1['above_line'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train = X.iloc[:75]
#y_train = y.iloc[:75]
#X_test = X.iloc[75:83]
#y_test = y.iloc[75:83]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

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
    'pts_reb+ast_ratio': [set1.iloc[:6]['PTS'].mean() / (set1.iloc[:6]['TRB'].mean() - set1.iloc[:6]['AST'].mean())],
    '3pa_fga_ratio': [set1.iloc[:6]['3PA'].mean() / (set1.iloc[:6]['FGA'].mean() - set1.iloc[:6]['3PA'].mean())],
    'PTS': [set1.iloc[:6]['PTS'].mean()],
    'Line': [18.5],
    'Opp_encoded' : [1],
    'LOC_@': [1],
    'Result_enc': [0],
    'Point_Diff': [-9]

})

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
X2 = set1.drop('Line', axis=1)
Y2 = set1['Line']

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)
model2 = RandomForestRegressor(n_estimators=100).fit(X2_train, Y2_train)

Y2_pred = model2.predict(X2_test)
print(accuracy_score(Y2_test, Y2_pred))